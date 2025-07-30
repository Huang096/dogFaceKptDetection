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

# 关键点语义名称
landmark_names = [
    "left_ear_tip",            # 0
    "right_ear_tip",           # 1
    "left_ear_upper",          # 2
    "right_ear_upper",         # 3
    "left_ear_base",           # 4
    "right_ear_base",          # 5
    "left_head_edge_top",      # 6
    "right_head_edge_top",     # 7
    "left_forehead",           # 8
    "right_forehead",          # 9
    "left_eyebrow_outer",      # 10
    "right_eyebrow_outer",     # 11
    "left_eyebrow_inner",      # 12
    "right_eyebrow_inner",     # 13
    "forehead_center_left",    # 14
    "forehead_center_right",   # 15
    "left_eye_outer",          # 16
    "right_eye_outer",         # 17
    "left_eye_center",         # 18
    "right_eye_center",        # 19
    "left_eye_inner",          # 20
    "right_eye_inner",         # 21
    "nose_left",               # 22
    "nose_right",              # 23
    "nose_top",                # 24
    "nose_center",             # 25
    "nose_bottom",             # 26
    "mouth_top_left",          # 27
    "mouth_top_right",         # 28
    "mouth_top_center",        # 29
    "mouth_left",              # 30
    "mouth_right",             # 31
    "mouth_bottom_left",       # 32
    "mouth_bottom_right",      # 33
    "mouth_bottom_center",     # 34
    "chin_upper",              # 35
    "chin_center",             # 36
    "jaw_left_upper",          # 37
    "jaw_right_upper",         # 38
    "jaw_left_mid",            # 39
    "jaw_right_mid",           # 40
    "jaw_left_lower",          # 41
    "jaw_right_lower",         # 42
    "neck_left",               # 43
    "neck_right",              # 44
    "neck_center_bottom"       # 45
]


# 输入目录
label_dirs = [
    "/Users/huangzheheng/Desktop/DogFLW/train/labels",
    "/Users/huangzheheng/Desktop/DogFLW/test/labels"
]

# 处理函数
def convert_landmarks_in_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    raw_landmarks = data.get("landmarks", [])
    if len(raw_landmarks) != 46:
        print(f"⚠️ Skipping {file_path}: expected 45 landmarks, got {len(raw_landmarks)}")
        return

    named_landmarks = []
    for idx, (x, y) in enumerate(raw_landmarks):
        named_landmarks.append({
            "id": idx,
            "name": landmark_names[idx],
            "x": x,
            "y": y
        })

    # 替换原数据中的 landmarks
    data["landmarks"] = named_landmarks

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated: {file_path}")

# 遍历两个目录
for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(label_dir, filename)
            convert_landmarks_in_file(file_path)
