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
import numpy as np

# ==== 路径配置 ====
img_dir = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedImg"
label_dir = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedLabel"
output_dir = "/Users/huangzheheng/Desktop/dogFaceKptDetection/output_vis"

os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# ==== 遍历图像 ====
for filename in os.listdir(img_dir):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    img_path = os.path.join(img_dir, filename)
    json_name = filename.replace(".jpg", ".json").replace(".png", ".json")
    json_path = os.path.join(label_dir, json_name)

    # 检查标签是否存在
    if not os.path.exists(json_path):
        print(f"❌ 缺失标签：{json_name}")
        continue

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 图像无法读取：{filename}")
        continue

    # 读取 JSON 中的 landmarks
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        landmarks = data["landmarks"]
    except Exception as e:
        print(f"❌ 标签读取失败：{json_name} — {e}")
        continue

    # 检查并绘制关键点
    has_invalid_point = False
    for pt in landmarks:
        x, y = pt.get("x"), pt.get("y")
        if x is None or y is None or not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or np.isnan(x) or np.isnan(y):
            print(f"❌ 非法关键点（文件: {json_name}）: {pt}")
            has_invalid_point = True
            break

    if has_invalid_point:
        print(f"🗑 删除异常样本：{filename} 和 {json_name}")
        # 删除图片和标签文件
        try:
            os.remove(img_path)
            os.remove(json_path)
        except Exception as e:
            print(f"⚠️ 删除文件出错：{e}")
        continue  # 跳过保存

    # 全部点合法，开始绘制
    for pt in landmarks:
        x, y = int(pt["x"]), int(pt["y"])
        cv2.circle(image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.putText(image, str(pt["id"]), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # 保存带标注的图像
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)
    print(f"✅ 已保存：{out_path}")