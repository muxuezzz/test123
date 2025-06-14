import json
import logging
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from src.pipeline import MultiPipelineSystem
from src.processors import RecInfer, RecInit, RecPostprocess, RecPreprocess
from src.utils import check_directory_readable, check_file_exists, preprocess_boxes

app = FastAPI()

# 全局变量
pipelines = None
args = None


class OCRConfig(BaseModel):
    det_model_path: str
    rec_model_path: str
    rec_labels_path: str = "ppocr_keys_v1.txt"
    use_cls: bool = False
    cls_model_path: str = None
    input_images_path: str = None
    infer_res_save_path: str = None
    det_image_save_path: str = None
    device_id: int = 0
    device_target: str = "310"
    parallel_num: int = 1
    timeout: int = 10
    do_profiling: bool = False
    rec_model_fixed_height: int = 32
    rec_model_min_width: int = 32
    rec_model_max_width: int = 4096
    rec_model_max_batch: int = 8


def load_config_from_file(config_path="src/demo/python/config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return OCRConfig(**config_dict)


def check_args(args):
    if not check_directory_readable(args.input_images_path):
        logging.error("Error! Please check the dataset path is set and valid.")
        raise NotADirectoryError(
            "Error! Please check the dataset path is set and valid."
        )

    if not check_file_exists(args.det_model_path):
        logging.error("Error! Please check the det model path is set and valid.")
        raise FileNotFoundError(
            "Error! Please check the det model path is set and valid."
        )

    if not check_directory_readable(args.rec_model_path) and not check_file_exists(
        args.rec_model_path
    ):
        logging.error("Error! Please check the rec model path is set and valid.")
        raise FileNotFoundError(
            "Error! Please check the rec model path is set and valid."
        )

    if not check_file_exists(args.rec_labels_path):
        logging.error(
            "Error! Please check the dictionary file of rec model path is set and valid."
        )
        raise FileNotFoundError(
            "Error! Please check the dictionary file of rec model path is set and valid."
        )

    if args.use_cls and not check_file_exists(args.cls_model_path):
        logging.error(
            "Error! Please check the dictionary file of cls model path is set and valid."
        )
        raise ValueError(
            "Error! Please check the dictionary file of cls model path is set and valid."
        )


@app.on_event("startup")
def startup_event():
    global pipelines, args
    args = load_config_from_file()
    check_args(args)

    rec_init_modules = [
        RecInit(args),
    ]
    rec_modules = [
        RecPreprocess(args),
        RecInfer(args),
        RecPostprocess(args),
    ]
    modules = rec_init_modules + rec_modules
    pipelines = MultiPipelineSystem(args.parallel_num, 32, modules)
    logging.info("Pipeline 初始化完成")


@app.post("/ocr/")
async def ocr(frame: UploadFile = File(...), boundingbox: np.array = Form(...)):
    """
    boundingbox: 期望格式为字符串，如 "x1,y1,w1,h1;x2,y2,w2,h2"
    """
    try:
        # 读取图片
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "图片解码失败"}

        boxes = preprocess_boxes(boundingbox)

        # 直接将img和boxes传入pipeline
        start = time.time()
        total_times = []
        rec_infer_time = 0
        results = []

        pipelines.send((img, boxes))
        _, metadata = pipelines.get(timeout=args.timeout)
        rec_infer_time += getattr(metadata, "rec_infer_time", 0)
        total_times.append(sum(getattr(metadata, "profiler", [0])))
        results.append(getattr(metadata, "result", None))
        total_size = len(boxes)

        total_time = time.time() - start
        msg = [
            f"average fps :{total_size / (total_time + 1e-6)}",
            f"total e2e time :{total_time} s",
            f"average latency time :{sum(total_times) / total_size * 1000 if total_size else 0} ms",
            f"rec per infer time :{rec_infer_time / total_size * 1000 if total_size else 0} ms",
        ]

        # 可选：保存推理结果
        if args.infer_res_save_path:
            os.makedirs(args.infer_res_save_path, exist_ok=True)
            with open(
                os.path.join(args.infer_res_save_path, "infer_time.txt"), "w"
            ) as f:
                f.write("\n".join(msg))

        return {"result": msg, "ocr_results": results}
    except Exception as e:
        logging.exception("OCR接口异常")
        return {"error": str(e)}
        logging.exception("OCR接口异常")
        return {"error": str(e)}
