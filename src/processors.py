#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: OCR e2e infer processors based on PaddleOCR3.0 on Ascend device
Author: MindX SDK
Create: 2022
History: NA
"""

import bisect
import glob
import logging
import math
import os
import stat
import time

import cv2
import numpy as np
from mindx.sdk import Tensor, base

from src.gear_matching import get_matched_gear, get_nearest_gear, get_shape_from_gear
from src.pipeline import MetaData, PipelineWorker
from src.utils import (
    boxes_from_bitmap,
    debug_var,
    decode_text,
    get_rotate_crop_image,
    get_upper_index_from_list,
    img_read,
    read_crooped_image,
)


class DetPreprocessDecodeResize(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        base.mx_init()
        dbnet = base.model(self.args.det_model_path, deviceId=self.args.device_id)
        self.gears = get_shape_from_gear(dbnet.model_gear())
        # self.imageProcessor = ImageProcessor(self.args.device_id)  # 初始化一个解码的dvpp对象
        del dbnet

    def process(self, image_path):
        name, _ = os.path.splitext(os.path.basename(image_path))
        if self.args.device_target == "310P":
            try:
                dvpp_image_src = self.imageProcessor.decode(
                    image_path, base.bgr
                )  # 获得解码输出后的Image类
                dvpp_image_src.to_host()
                image_src = np.array(dvpp_image_src.to_tensor())[
                    0, : dvpp_image_src.original_height, : dvpp_image_src.original_width
                ]
                del dvpp_image_src
            except Exception as e:  # 捕获所有继承自 Exception 的异常
                logging.warning(
                    f"DVPP not available or failed ({e}), using OpenCV instead!"
                )
                # 使用 logging.exception 会打印完整的 traceback，对于调试非常有用
                logging.exception("Detailed error info for DVPP failure:")
                image_src = img_read(image_path)

        else:
            image_src = img_read(image_path)
        h, w = image_src.shape[:2]
        metadata = MetaData()
        metadata.name = name
        metadata.h = h
        metadata.w = w
        metadata.image_path = image_path
        metadata.image_src = image_src

        limit_side_len = 960
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        image = cv2.resize(image_src, (int(resize_w), int(resize_h)))

        return image, metadata


class DetPreprocessNormalize(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        self.scale = 1.0 / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def process(self, image, metadata):
        image = (image.astype("float32") * self.scale - self.mean) / self.std

        return image, metadata


class DetPreprocessTranspose(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        pass

    def process(self, image, metadata):
        image2 = image.transpose(2, 0, 1)
        image2 = np.expand_dims(image2, 0)  # [B, C, H, W]

        image3 = np.ascontiguousarray(image2).astype(np.float32)
        return image3, metadata


class DetInfer(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        base.mx_init()
        self.dbnet = base.model(self.args.det_model_path, deviceId=self.args.device_id)

        # for pre-heating
        h, w = 1280, 1280
        gears = get_shape_from_gear(self.dbnet.model_gear())
        self.gears = gears
        gear = get_nearest_gear(w, h, gears)
        image = np.zeros((1, 3, gear[1], gear[0])).astype(np.float32)
        inputs = [Tensor(image)]
        self.dbnet.infer(inputs)

    def process(self, image, metadata):
        inputs = []
        metadata.det_infer_time = 0

        resize_w, resize_h = image.shape[3], image.shape[2]

        # pad input to match model gears
        gear = get_matched_gear(resize_w, resize_h, self.gears)
        if gear != (resize_w, resize_h):
            image_pad = np.zeros((1, 3, gear[1], gear[0]), dtype=np.float32)
            image_pad[:, :, :resize_h, :resize_w] = image
            image = image_pad

        inputs.append(Tensor(image))
        det_infer_start = time.time()
        output = self.dbnet.infer(inputs)
        if not output:
            output = self.dbnet.infer(inputs)
        infer_res = output[0]
        infer_res.to_host()
        metadata.det_infer_time = time.time() - det_infer_start
        res_infer = np.array(infer_res)

        # crop the padded area
        res_infer = res_infer[:, 0, :resize_h, :resize_w]

        return res_infer, metadata


class DetPostproces(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        base.mx_init()

    def process(self, res_infer, metadata):
        rec_list = []
        crnn_image_list = []
        thresh = 0.3
        segmentation = res_infer > thresh  # 对概率图进行固定阈值处理，得到分割图

        boxes_batch = []
        scores_batch = []
        score_thresh = 0
        boxes, scores = boxes_from_bitmap(
            res_infer, segmentation, metadata.w, metadata.h
        )
        boxes_batch.append(boxes)
        scores_batch.append(scores)
        for i in range(len(scores)):
            if scores[i] <= score_thresh:
                continue
            if boxes[i][1][1] >= boxes[i][3][1] or boxes[i][3][0] >= boxes[i][1][0]:
                continue
            rec_list.append(
                list(
                    map(
                        str,
                        [
                            boxes[i][1][0],
                            boxes[i][1][1],
                            boxes[i][2][0],
                            boxes[i][2][1],
                            boxes[i][3][0],
                            boxes[i][3][1],
                            boxes[i][0][0],
                            boxes[i][0][1],
                        ],
                    )
                )
            )
            points = np.array(
                [
                    [boxes[i][0][0], boxes[i][0][1]],
                    [boxes[i][1][0], boxes[i][1][1]],
                    [boxes[i][2][0], boxes[i][2][1]],
                    [boxes[i][3][0], boxes[i][3][1]],
                ],
                dtype=np.float32,
            )
            crnn_image = get_rotate_crop_image(metadata.image_src, points)
            crnn_image_list.append(crnn_image)
            print(f"crnn_image {i}:", crnn_image.shape)
            # debug_var(crnn_image)
            if self.args.det_image_save_path:
                cv2.imwrite(
                    os.path.join(
                        self.args.det_image_save_path, f"{metadata.name}_{i}.jpg"
                    ),
                    crnn_image,
                )
        print("rec_list:")
        debug_var(rec_list)
        print("metadata:")
        debug_var(metadata)

        metadata.rec_list = rec_list
        del metadata.image_src
        return crnn_image_list, metadata


class ClsPreProcess(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args
        self.cls_image_shape = (3, 48, 192)
        self.batchlist = []

    def init(self):
        base.mx_init()
        cls = base.model(self.args.cls_model_path, deviceId=self.args.device_id)
        gears = get_shape_from_gear(cls.model_gear())
        self.batchlist = sorted(gears["batch_size"].tolist())
        del cls

    def get_batchsize_list(self, img_list_size):
        index = bisect.bisect_right(self.batchlist, img_list_size)
        return self.batchlist[min(index, len(self.batchlist) - 1)]

    def process(self, img_list, metadata):
        if not img_list:
            metadata.inds = []
            return {}, {}, metadata
        tensor_list = []
        actual_sizes = []
        index = 0
        n = len(img_list)
        while index < n:
            batchsize = self.get_batchsize_list(n - index)
            tensor = []
            for img_id in range(min(batchsize, n - index)):
                # 前处理
                img = img_list[img_id + index]
                h, w, c = img.shape
                input_c, input_h, input_w = self.cls_image_shape
                ratio = w / float(h)
                if math.ceil(input_h * ratio) > input_w:
                    resized_w = input_w
                else:
                    resized_w = math.ceil(input_h * ratio)
                image = cv2.resize(img, (resized_w, input_h))
                resized_image = image.transpose((2, 0, 1)) / 255
                padding_im = np.zeros((input_c, input_h, input_w), dtype=np.float32)
                padding_im[:, :, 0:resized_w] = resized_image
                padding_im -= 0.5
                padding_im /= 0.5
                tensor.append(np.ascontiguousarray(padding_im))
            actual_sizes.append(len(tensor))
            tensor = np.array(tensor)
            if batchsize - actual_sizes[-1]:
                tensor = np.concatenate(
                    (
                        tensor,
                        np.zeros(
                            (batchsize - actual_sizes[-1], input_c, input_h, input_w),
                            dtype=np.float32,
                        ),
                    )
                )
            tensor_list.append(tensor)
            index += actual_sizes[-1]
        metadata.actual_sizes = actual_sizes
        return img_list, tensor_list, metadata


class ClsProcess(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args
        self.cls = None
        self.thresh = 0.9

    def init(self):
        base.mx_init()
        self.cls = base.model(self.args.cls_model_path, deviceId=self.args.device_id)

    def process(self, img_list, tensor_list, metadata):
        if not img_list:
            metadata.inds = []
            return [], metadata
        img_list_index = 0
        for index, img in enumerate(tensor_list):
            # 推理
            inputs = [Tensor(img)]
            output = self.cls.infer(inputs)
            output = output[0]
            output.to_host()
            output = np.array(output)

            # 后处理
            # fix_index为单个batch中的index
            for fix_index in range(metadata.actual_sizes[index]):
                if output[fix_index, 1] > self.thresh:
                    img_list[img_list_index + fix_index] = cv2.rotate(
                        img_list[img_list_index + fix_index], cv2.ROTATE_180
                    )
            img_list_index += metadata.actual_sizes[index]
        return img_list, metadata


class RecPreprocess(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args
        self.crnn_list = []
        self.is_dynamic = False

    def init(self):
        base.mx_init()
        if os.path.isdir(self.args.rec_model_path):
            om_list = sorted(glob.glob(os.path.join(self.args.rec_model_path, "*.om")))
            if not om_list:
                raise FileNotFoundError("om model does not exist in model path.")
            crnn_list = []
            gear_list = []
            for om in om_list:
                model = base.model(om, deviceId=self.args.device_id)
                gear = model.model_gear()
                gear_list.append(np.array(gear))
                del model
            batchsizes = [gear[0][0] for gear in gear_list]

            self.crnn_list = sorted(
                crnn_list, key=lambda x: batchsizes[crnn_list.index(x)]
            )
            self.batchsizes = sorted(batchsizes)
            self.gears = get_shape_from_gear(gear)
            heights = list(set(self.gears["height"]))
            if len(heights) > 1:
                raise NotImplementedError(
                    "om model input shapes should have same height!"
                )
            self.height = heights[0]
            self.rec_model_min_width = min(self.gears["width"])
            self.rec_model_max_width = max(self.gears["width"])
        else:
            self.is_dynamic = True
            self.height = self.args.rec_model_fixed_height
            self.rec_model_min_width = self.args.rec_model_min_width
            self.rec_model_max_width = self.args.rec_model_max_width

    def process(self, crnn_image_list, metadata):
        resize_widths = []
        inds = []
        images = []
        # if no image, then skip.
        if not crnn_image_list:
            metadata.inds = []
            return [], metadata
        # preprocess for recognition
        for ind, img in enumerate(crnn_image_list):
            h, w = img.shape[:2]
            resized_w = max(
                min(math.ceil(w * self.height / h), self.rec_model_max_width),
                self.rec_model_min_width,
            )
            resize_widths.append(resized_w)
            inds.append(ind)
            resized_image = cv2.resize(img, (resized_w, self.height))
            images.append(resized_image)
        sorted_zip = sorted(
            zip(inds, resize_widths, images), key=lambda x: x[1], reverse=True
        )

        metadata.inds, _, _ = zip(*sorted_zip)

        padding_ims = []
        actual_sizes = []
        print("sorted_zip")
        print(type(sorted_zip))
        print(len(sorted_zip))
        print("batchsizes:", self.batchsizes)
        while len(sorted_zip):
            if not self.is_dynamic:  # False
                current_batch = get_upper_index_from_list(
                    len(sorted_zip), self.batchsizes
                )
                batch_size = self.batchsizes[current_batch]
                actual_size = min(
                    len(sorted_zip), batch_size, self.args.rec_model_max_batch
                )
                gear_widths = self.gears["width"].to_list()
                resize_shape = gear_widths[
                    get_upper_index_from_list(sorted_zip[0][1], gear_widths)
                ]
                # padding_im = np.zeros((batch_size, 3, self.height, resize_shape), dtype=np.float32)
                padding_im = np.zeros(
                    (actual_size, 3, self.height, resize_shape), dtype=np.float32
                )
            else:
                actual_size = min(len(sorted_zip), self.args.rec_model_max_batch)
                resize_shape = max(resize_widths)
                padding_im = np.zeros(
                    (actual_size, 3, self.height, resize_shape), dtype=np.float32
                )
            print("actual_size:", actual_size)
            print("padding_im.shape:", padding_im.shape)
            for i in range(actual_size):
                img = sorted_zip[i][2]
                resized_image = img.astype("float32")
                resized_image = resized_image.transpose((2, 0, 1)) / 255
                resized_image -= 0.5
                resized_image /= 0.5
                resized_image = np.ascontiguousarray(resized_image.astype(np.float32))
                padding_im[i, :, :, : img.shape[1]] = resized_image
                # debug_var(padding_im)
                print(f"padding_im {i}:")
                print(padding_im[i].shape)
            padding_ims.append(padding_im)
            actual_sizes.append(actual_size)
            sorted_zip = sorted_zip[actual_size:]
        metadata.actual_sizes = actual_sizes
        return padding_ims, metadata


class RecInfer(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args
        self.crnn_list = []
        self.is_dynamic = False

    def init(self):
        base.mx_init()
        if os.path.isdir(self.args.rec_model_path):
            om_list = sorted(glob.glob(os.path.join(self.args.rec_model_path, "*.om")))
            if not om_list:
                raise FileNotFoundError("om model does not exist in model path.")
            crnn_list = []
            gear_list = []
            for om in om_list:
                model = base.model(om, deviceId=self.args.device_id)
                gear = model.model_gear()
                gear_list.append(np.array(gear))
                crnn_list.append(model)
            batchsizes = [gear[0][0] for gear in gear_list]

            self.crnn_list = sorted(
                crnn_list, key=lambda x: batchsizes[crnn_list.index(x)]
            )
            self.batchsizes = sorted(batchsizes)
        else:
            self.is_dynamic = True
            self.crnn = base.model(
                self.args.rec_model_path, deviceId=self.args.device_id
            )

    def process(self, padding_ims, metadata):
        metadata.rec_infer_time = 0
        b_logits_list = []
        for idx, padding_im in enumerate(padding_ims):
            inputs = [Tensor(padding_im)]
            rec_infer_start = time.time()
            if not self.is_dynamic:  # False
                current_batch = self.batchsizes.index(len(padding_im))
                actual_size = metadata.actual_sizes[idx]
                output = self.crnn_list[current_batch].infer(inputs)
            else:
                actual_size = metadata.actual_sizes[idx]
                output = self.crnn.infer(inputs)
            infer_res = output[0]
            infer_res.to_host()
            metadata.rec_infer_time += time.time() - rec_infer_start
            b_logits = np.array(infer_res)[:actual_size]
            b_logits_list.append(b_logits)
        return b_logits_list, metadata


class RecPostprocess(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        labels = [""]
        with open(
            os.path.realpath(self.args.rec_labels_path), "r", encoding="utf-8"
        ) as f:
            labels.extend(f.readlines())
        labels.append(" ")
        self.labels = labels

    def process(self, b_logits_list, metadata):
        args = self.args
        rec_list = metadata.rec_list
        inds = list(metadata.inds)
        inds.reverse()
        for b_logits in b_logits_list:
            for idx, logits in enumerate(b_logits):
                pred = logits
                pred = np.expand_dims(pred, axis=0)
                pred_text = decode_text(self.labels, pred, [pred.shape[1]])[0]
                index = inds.pop()
                if pred_text.strip():
                    rec_list[index].append(pred_text)
                else:
                    # 识别不到内容
                    rec_list[index].append("###")

        infer_res_name = f"infer_img_{metadata.name.split('_')[-1]}.txt"
        if args.infer_res_save_path:
            flags, modes = (
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP,
            )
            with os.fdopen(
                os.open(
                    os.path.join(args.infer_res_save_path, infer_res_name), flags, modes
                ),
                "w",
            ) as f:
                if not rec_list:
                    f.write("")
                for res in rec_list:
                    f.write(",".join(res) + "\n")
        return infer_res_name, metadata


## new
class RecInit(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        pass

    #     labels = ['']
    #     with open(os.path.realpath(self.args.rec_labels_path), 'r', encoding="utf-8") as f:
    #         labels.extend(f.readlines())
    #     labels.append(' ')
    #     self.labels = labels

    def process(self, images_dir):
        metadata = MetaData()
        metadata.name = "test"

        rec_list = []
        crnn_image_list = []

        image_list = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(
            os.path.join(images_dir, "*.jpeg")
        )
        image_list.sort()

        for i in range(len(image_list)):
            crnn_image = read_crooped_image(image_list[i])
            crnn_image_list.append(crnn_image)
            print(f"crnn_image {i}:", crnn_image.shape)
            rec_list.append(list(map(str, [os.path.basename(image_list[i])])))

            # debug_var(crnn_image)
        print("rec_list:")
        debug_var(rec_list)
        print("metadata:")
        debug_var(metadata)

        metadata.rec_list = rec_list
        return crnn_image_list, metadata


class Image_Init(PipelineWorker):
    def __init__(self, args):
        super().__init__(args.do_profiling)
        self.args = args

    def init(self):
        pass

    def cut_by_boxes_class(self, image, boxes, c):
        """
        根据检测框裁剪图像中指定类别的区域。

        参数:
            image (numpy.ndarray): 原始图像数据。
            boxes (numpy.ndarray): 检测框数组，形状为 (n, 7)，
                                  包含 [x_min, y_min, x_max, y_max, score, class_id, track_id]。
            c (int): 要裁剪的类别 ID。

        返回:
            list: 裁剪后的图像区域列表 (numpy.ndarray)。
        """
        cropped_images = []
        track_ids = []  # 新增：用于存储 track_id 的列表
        for box in boxes:
            x_min, y_min, x_max, y_max, _, class_id, track_id = box
            if int(class_id) == c:  # 确保 class_id 是整数类型
                x_min, y_min, x_max, y_max = map(
                    int, [x_min, y_min, x_max, y_max]
                )  # 确保坐标是整数
                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_images.append(cropped_image)
                track_ids.append(track_id)  # 新增：将 track_id 添加到列表中
        return cropped_images, track_ids  # 修改：返回裁剪后的图像和 track_id 列表

    def cut_by_boxes(self, image, boxes):
        """
        根据检测框裁剪图像中指定类别的区域。

        参数:
            image (numpy.ndarray): 原始图像数据。
            boxes (list): 检测框数组，形状为 (n, 7)，
                                  包含 [x_min, y_min, x_max, y_max, score, class_id, track_id]。

        返回:
            list: 裁剪后的图像区域列表 (numpy.ndarray)。
        """
        cropped_images = []
        track_ids = []  # 新增：用于存储 track_id 的列表
        for box in boxes:
            x_min, y_min, x_max, y_max, _, _, track_id = box
            x_min, y_min, x_max, y_max = map(
                int, [x_min, y_min, x_max, y_max]
            )  # 确保坐标是整数
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped_image)
            track_ids.append(track_id)  # 新增：将 track_id 添加到列表中
        return cropped_images, track_ids  # 修改：返回裁剪后的图像和 track_id 列表

    def process(self, image, boxes, c):
        """
        处理图像和检测框，提取指定类别的裁剪区域。

        参数:
            image (numpy.ndarray): 原始图像数据。
            boxes (numpy.ndarray): 检测框数组，形状为 (n, 7)，
                                  包含 [x_min, y_min, x_max, y_max, score, class_id, track_id]。
            c (int): 要裁剪的类别ID.

        返回:
            tuple: 包含裁剪后的图像列表和元数据的元组。
                   - crnn_image_list (list): 裁剪后的图像区域列表 (numpy.ndarray)。
                   - metadata (MetaData): 包含处理信息的元数据对象。
        """
        metadata = MetaData()
        metadata.name = "cropped_images"
        crnn_image_list, track_ids = self.cut_by_boxes(
            image, boxes
        )  # 修改：接收返回的 track_ids
        metadata.rec_list = [
            f"crop_{i}" for i in range(len(crnn_image_list))
        ]  # 创建一个虚拟的 rec_list
        metadata.track_ids = track_ids  # 新增：将 track_ids 存储在 metadata 中
        return crnn_image_list, metadata
