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

import cv2
import json

# 输入路径
image_path = "/Users/huangzheheng/Desktop/DogFLW/train/images/n02085620_11948.png"
json_path = "/Users/huangzheheng/Desktop/DogFLW/train/labels/n02085620_11948.json"
output_path = "/Users/huangzheheng/Desktop/dogFaceKptDetection/labeledImg/n02085620_11948_labeled.jpg"

# 读取 landmark 数据
with open(json_path, "r") as f:
    data = json.load(f)
    landmarks = data["landmarks"]

# 读取图像
img = cv2.imread(image_path)

# 可视化关键点
for pt in landmarks:
    idx = pt["id"]
    x, y = int(pt["x"]), int(pt["y"])

    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Green dot

    # 偏移映射
    offset_map = {
        41: (6, 14), 42: (-12, -10), 35: (10, 10), 27: (12, -10)
    }
    dx, dy = offset_map.get(idx, (6, -10))

    text = str(idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    color = (0, 0, 255)
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 白色背景 + 红色文字
    cv2.rectangle(img, (x + dx, y + dy - text_h), (x + dx + text_w, y + dy), (255, 255, 255), -1)
    cv2.putText(img, text, (x + dx, y + dy), font, font_scale, color, thickness, cv2.LINE_AA)

# 保存结果
cv2.imwrite(output_path, img)
print(f"✔️ 可视化图像已保存：{output_path}")
