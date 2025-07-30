# Copyright 2025 huangzheheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cv2
import argparse

# —— 新增：从命令行读取路径 —— 
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',        required=True)
parser.add_argument('--label_dir',        required=True)
parser.add_argument('--output_img_dir',   required=True)
parser.add_argument('--output_label_dir', required=True)
args = parser.parse_args()

image_dir      = args.image_dir
label_dir      = args.label_dir
output_img_dir = args.output_img_dir
output_label_dir = args.output_label_dir

# —— 以下保持不变，只用变量 —— 
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
target_size = 224

# 创建输出文件夹
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

target_size = 224

def process_and_save(image_path, label_path, output_img_path, output_label_path):
    with open(label_path, "r") as f:
        data = json.load(f)

    # 检查 bounding_boxes 合法性
    if "bounding_boxes" not in data or len(data["bounding_boxes"]) != 4:
        print(f"⚠️ Skipping {label_path}: missing or incomplete bounding_boxes")
        return

    try:
        bbox = list(map(float, data["bounding_boxes"]))
    except ValueError:
        print(f"⚠️ Skipping {label_path}: bounding_boxes contains invalid values")
        return

    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping {image_path}: failed to read image")
        return

    h, w = image.shape[:2]

    # 检查 bbox 越界
    if x_min >= x_max or y_min >= y_max or x_max > w or y_max > h:
        print(f"⚠️ Skipping {label_path}: bounding box out of image bounds")
        return

    cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    resized = cv2.resize(cropped, (target_size, target_size))

    scale_x = target_size / width
    scale_y = target_size / height

    adjusted_landmarks = []
    for lm in data.get("landmarks", []):
        new_x = (lm["x"] - x_min) * scale_x
        new_y = (lm["y"] - y_min) * scale_y
        adjusted_landmarks.append({
            "id": lm["id"],
            "name": lm["name"],
            "x": round(new_x, 2),
            "y": round(new_y, 2)
        })

    # 保存新图像和标签
    cv2.imwrite(output_img_path, resized)
    with open(output_label_path, "w") as f:
        json.dump({"landmarks": adjusted_landmarks}, f, indent=2)

# 遍历所有图像和 JSON
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, f"{name}.json")
        out_img_path = os.path.join(output_img_dir, f"{name}.jpg")
        out_label_path = os.path.join(output_label_dir, f"{name}.json")

        if os.path.exists(label_path):
            process_and_save(img_path, label_path, out_img_path, out_label_path)
