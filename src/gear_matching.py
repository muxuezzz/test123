#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
Description: gear matching strategies.
Author: MindX SDK
Create: 2022
History: NA
"""

import numpy as np
import pandas as pd

eps = 1e-6


def get_shape_from_gear(gear):
    first = gear[0]
    if len(first) != 4:
        raise ValueError("model input dim must be 4")
    if 1 <= first[1] <= 3:
        nchw = True
    elif 1 <= first[3] <= 3:
        nchw = False
    else:
        raise ValueError("model channel number must be in [1,3]")

    wh_list = []
    if nchw:
        for item in gear:
            wh_list.append([item[0], item[2], item[3], item[1]])
    else:
        for item in gear:
            wh_list.append(item)

    wh_array = np.array(wh_list)

    df = pd.DataFrame(wh_array, columns=["batch_size", "height", "width", "channels"])
    df["aspect_ratios"] = df["width"] * 1.0 / (df["height"] + eps)
    df["wh_sums"] = df["width"] + df["height"]
    df["wh_product"] = df["width"] * df["height"]

    return df


def get_matched_gear(width, height, df):
    in_range = df[(df["width"] > width) & (df["height"] > height)]
    if len(in_range) > 0:
        chosen = in_range["wh_product"].argmin()
        chosen_item = in_range.iloc[chosen]
    else:
        chosen = df["wh_product"].argmax()
        chosen_item = df.iloc[chosen]
    return int(chosen_item["width"]), int(chosen_item["height"])


def get_nearest_gear(width, height, df, thresh=0.0):
    aspect_ratio_current = width / (height + eps)

    df["wh_sum_diffs"] = abs(df["wh_sums"] - (height + width))
    df["aspect_ratio_diffs"] = abs(df["aspect_ratios"] - aspect_ratio_current)
    in_range = df[df["aspect_ratio_diffs"] < thresh]
    if len(in_range) > 0:
        chosen = in_range["wh_sum_diffs"].argmin()
        chosen_item = in_range.iloc[chosen]
    else:
        chosen = df["aspect_ratio_diffs"].argmin()
        chosen_item = df.iloc[chosen]
    return int(chosen_item["width"]), int(chosen_item["height"])
