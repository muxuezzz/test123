#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: utils of processors
Author: MindX SDK
Create: 2022
History: NA
"""
import os

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import numpy as np

def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    '''

    bitmap = _bitmap.reshape(_bitmap.shape[1], _bitmap.shape[2])
    pred = pred.reshape(pred.shape[1], pred.shape[2])

    height, width = bitmap.shape
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_candidates = 1000
    num_contours = min(len(contours), max_candidates)
    boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
    scores = np.zeros((num_contours,), dtype=np.float32)

    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        min_size = 3
        if sside < min_size:
            continue
        points = np.array(points)
        score = box_score_fast(pred, points.reshape(-1, 2))
        box_thresh = 0.5
        if box_thresh > score:
            continue

        bbox = unclip(points).reshape(-1, 1, 2)
        bbox, sside = get_mini_boxes(bbox)
        if sside < min_size + 2:
            continue
        bbox = np.array(bbox)

        bbox[:, 0] = np.clip(
            np.round(bbox[:, 0] / width * dest_width), 0, dest_width)
        bbox[:, 1] = np.clip(
            np.round(bbox[:, 1] / height * dest_height), 0, dest_height)
        boxes[index, :, :] = bbox.astype(np.int16)
        scores[index] = score
    return boxes, scores


def unclip(box, unclip_ratio=2.0):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    _offset = pyclipper.PyclipperOffset()
    _offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(_offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_one = 0
        index_four = 1
    else:
        index_one = 1
        index_four = 0
    if points[3][1] > points[2][1]:
        index_two = 2
        index_three = 3
    else:
        index_two = 3
        index_three = 2

    box = [points[index_one], points[index_two],
           points[index_three], points[index_four]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    x_min = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    x_max = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    y_min = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    y_max = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - x_min
    box[:, 1] = box[:, 1] - y_min
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[y_min:y_max + 1, x_min:x_max + 1].astype(np.float32), mask)[0]


def get_rotate_crop_image(img, points):
    '''
    warpperspective an area into rectangle from img.
    '''
    if len(points) != 4:
        raise ValueError("shape of points must be 4*2")
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / (dst_img_width + 1e-6) >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def read_crooped_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取文件：{img_path}")
        return img
    except Exception as e:
        print(f"读取时发生错误：{e}")
        return None


def decode_text(character, text_index, length):
    """ convert text-index into text-label. """
    texts = []
    for index, l in enumerate(length):
        t = text_index[index, :]
        char_list = []
        for i in range(l):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                char_list.append(character[t[i]].strip())
        text = ''.join(char_list)
        texts.append(text)
    return texts


def img_read(path):
    path = os.path.realpath(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"File not found:{path}")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_upper_index_from_list(num, batchsizes):
    if num in batchsizes:
        return batchsizes.index(num)
    elif num > max(batchsizes):
        return len(batchsizes) - 1
    else:
        batchsizes_new = batchsizes.copy()
        batchsizes_new.append(num)
        batchsizes_new = sorted(batchsizes_new)
        return batchsizes_new.index(num)


def check_file_exists(filepath: str) -> bool:
    filepath = os.path.realpath(filepath)
    if not os.path.isfile(filepath):
        return False
    return True


def check_directory_readable(pathname: str) -> bool:
    pathname = os.path.realpath(pathname)
    if not isinstance(pathname, str) or not pathname:
        return False
    if not os.path.exists(pathname):
        return False
    if os.path.isfile(pathname):
        return False
    if os.path.islink(pathname):
        return False
    if not os.access(pathname, os.R_OK):
        return False
    return True


#################
def preprocess_boxes(boxes: np.ndarray, c: int) -> list[list[int]]:
    return [list(map(int, listtmp)) for listtmp in boxes.tolist() if int(listtmp[5]) == c]
    # pas

def debug_var(var):
    """
    打印变量的值、类型、形状（shape）以及可选的长度（len）信息。
    
    参数：
        var: 可以是任意类型的变量，例如数值、列表、字典或 numpy 数组等。
    输出：
        - Value: 变量的具体值
        - Type: 变量的数据类型
        - Shape (如果存在): 如果变量有 shape 属性（如 numpy 数组），则显示形状
        - Length (如果存在): 如果变量可计算长度（如列表、元组、字典等），则显示元素数量或键的数量
    """
    
    print("Debug Info:")
    print(f"Value: {var}")         # 输出值
    print(f"Type: {type(var)}")    # 输出类型
    
    try:
        # 尝试获取 shape 属性（针对 numpy 数组、张量等）
        shape = var.shape
        print(f"Shape: {shape}")
    except AttributeError:
        pass  # 无 shape 属性时忽略
    
    try:
        # 尝试计算长度（列表/元组的元素数量，字典的键的数量）
        length = len(var)
        print(f"Length: {length}")
    except (TypeError, AttributeError):
        pass  # 无法计算长度时忽略
    
    print("------")                # 分隔符