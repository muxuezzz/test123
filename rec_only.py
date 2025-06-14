#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: OCR e2e infer demo based on PaddleOCR3.0 on Ascend device
Author: MindX SDK
Create: 2022
History: NA
"""

import argparse
import ast
import glob
import logging
import os
import shutil
import stat
import time
from multiprocessing import Process

# from tqdm import tqdm

from src.pipeline import MultiPipelineSystem
from src.processors import DetPreprocessDecodeResize, DetPreprocessNormalize, DetPreprocessTranspose, DetInfer, \
    DetPostproces, RecInfer, RecPreprocess, RecPostprocess, ClsProcess, ClsPreProcess, RecInit
from src.utils import check_directory_readable, check_file_exists

logging.getLogger().setLevel(logging.INFO)


def save_path_init(opt):
    if opt.infer_res_save_path:
        if os.path.exists(opt.infer_res_save_path):
            shutil.rmtree(opt.infer_res_save_path)
        os.makedirs(opt.infer_res_save_path, 0o750)
    if opt.det_image_save_path:
        if os.path.exists(opt.det_image_save_path):
            shutil.rmtree(opt.det_image_save_path)
        os.makedirs(opt.det_image_save_path, 0o750)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model_path', type=str, required=True)
    parser.add_argument('--rec_model_path', type=str, required=True)
    parser.add_argument('--rec_labels_path', type=str, required=False, default='ppocr_keys_v1.txt')

    parser.add_argument('--use_cls', type=ast.literal_eval, required=False, default=False, choices=[True, False])
    parser.add_argument('--cls_model_path', type=str, required=False, default=None)

    parser.add_argument('--input_images_path', type=str, required=True)
    parser.add_argument('--infer_res_save_path', type=str, required=False, default=None)
    parser.add_argument('--det_image_save_path', type=str, required=False, default=None)
    parser.add_argument('--device_id', type=int, required=False, default=0)
    parser.add_argument('--device_target', type=str, required=False, default="310", choices=["310P", "310"])
    parser.add_argument('--parallel_num', type=int, required=False, default=1)
    parser.add_argument('--timeout', type=int, required=False, default=None)
    parser.add_argument('--do_profiling', type=ast.literal_eval, required=False,
                        default=False, choices=[True, False])
    parser.add_argument('--rec_model_fixed_height', type=int, required=False, default=32)
    parser.add_argument('--rec_model_min_width', type=int, required=False, default=32)
    parser.add_argument('--rec_model_max_width', type=int, required=False, default=4096)
    parser.add_argument('--rec_model_max_batch', type=int, required=False, default=8)
    args = parser.parse_args()
    return args


def check_args(args):
    if not check_directory_readable(args.input_images_path):
        logging.error('Error! Please check the dataset path is set and valid.')
        raise NotADirectoryError('Error! Please check the dataset path is set and valid.')

    if not check_file_exists(args.det_model_path):
        logging.error('Error! Please check the det model path is set and valid.')
        raise FileNotFoundError('Error! Please check the det model path is set and valid.')

    if not check_directory_readable(args.rec_model_path) and not check_file_exists(args.rec_model_path):
        logging.error('Error! Please check the rec model path is set and valid.')
        raise FileNotFoundError('Error! Please check the rec model path is set and valid.')

    if not check_file_exists(args.rec_labels_path):
        logging.error('Error! Please check the dictionary file of rec model path is set and valid.')
        raise FileNotFoundError('Error! Please check the dictionary file of rec model path is set and valid.')

    if args.use_cls and not check_file_exists(args.cls_model_path):
        logging.error('Error! Please check the dictionary file of cls model path is set and valid.')
        raise ValueError('Error! Please check the dictionary file of cls model path is set and valid.')


def main():
    # 参数解析
    args = parse_args()
    check_args(args)
    # model and save path init
    # save_path_init(args)
    # image_list = glob.glob(os.path.join(args.input_images_path, '*.jpg')) + \
    #              glob.glob(os.path.join(args.input_images_path, '*.jpeg'))
    # if not image_list:
    #     raise FileNotFoundError("no jpg images in input path.")
    # total_size = len(image_list)

    # det_modules = [
    #     DetPreprocessDecodeResize(args),
    #     DetPreprocessNormalize(args),
    #     DetPreprocessTranspose(args),
    #     DetInfer(args),
    #     DetPostproces(args),
    # ]
    # cls_modules = [
    #     ClsPreProcess(args),
    #     ClsProcess(args)
    # ]

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
    image_dir = r'/home/ocr-v6/mindxsdk-mxocr/src/demo/data/License'

    def send(pipelines, image_dir):
        pipelines.send(image_dir)

        # for idx, image_path in enumerate(image_list):
        #     pipelines.send(image_path)


    start = time.time()
    send_p = Process(target=send, args=(pipelines, image_dir))
    send_p.start()

    det_infer_time = 0
    rec_infer_time = 0
    total_times = []
    total_size = 1
    for _ in range(total_size):
        print("正式")
        try:
            _, metadata = pipelines.get(timeout=args.timeout)
        except:
            logging.error("Timeout! An error may happended during processing. main process will end.")
            pipelines.end()
            exit(-1)
        # det_infer_time += metadata.det_infer_time
        rec_infer_time += metadata.rec_infer_time
        total_times.append(sum(metadata.profiler))

    total_time = time.time() - start
    msg = [
        f"average fps :{total_size / (total_time + 1e-6)}",
        f"total e2e time :{total_time} s",
        f"average latency time :{sum(total_times) / total_size * 1000} ms",
        # f"det per infer time :{det_infer_time / total_size * 1000} ms",
        f"rec per infer time :{rec_infer_time / total_size * 1000} ms"
    ]
    logging.info("\n".join(msg))
    if args.infer_res_save_path:
        flags, modes = os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
        with os.fdopen(os.open(os.path.join(args.infer_res_save_path, 'infer_time.txt'), flags, modes), 'w') as f:
            f.write("\n".join(msg))
    send_p.join()
    pipelines.end()


if __name__ == '__main__':
    main()
