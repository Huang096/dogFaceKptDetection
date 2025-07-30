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

# 从你终端打印结果中整理出所有无效 label 的完整路径：
invalid_labels = [
    "n02109961_4619", "n02110627_13020", "n02109047_5588", "n02107683_34",
    "n02106662_320", "n02112137_2220", "n02107683_4410", "n02110627_11422",
    "n02106030_15169", "n02111500_6769", "n02112137_347", "n02101388_2833",
    "n02113624_7986", "n02113799_2600", "n02107683_2305", "n02113624_8444",
    "n02115913_1578", "n02107683_2059", "n02113624_300", "n02107908_4030",
    "n02106166_3882", "n02108422_2111", "n02108000_3299", "n02106382_1016",
    "n02111129_970", "n02107574_986", "n02112137_8410", "n02113799_7258",
    "n02105855_9277", "n02111277_6773", "n02108551_2550", "n02112350_177",
    "n02107312_398", "n02112706_220", "n02113799_273", "n02108000_1901",
    "n02108551_1705", "n02113799_5514", "n02107683_1076", "n02092339_4169",
    "n02108422_3273", "n02111500_5525", "n02109047_11184", "n02112137_11858",
    "n02106030_5915", "n02109525_2681", "n02109047_1480", "n02113799_5986"
]

label_dir = "/Users/huangzheheng/Desktop/DogFLW/train/labels"
image_dir = "/Users/huangzheheng/Desktop/DogFLW/train/images"

deleted_count = 0

for name in invalid_labels:
    label_path = os.path.join(label_dir, name + ".json")
    image_path = os.path.join(image_dir, name + ".png")

    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"🗑 Deleted JSON: {label_path}")
    else:
        print(f"⚠️ JSON not found: {label_path}")

    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"🗑 Deleted PNG : {image_path}")
    else:
        print(f"⚠️ PNG not found: {image_path}")

print("✅ 清理完毕")
