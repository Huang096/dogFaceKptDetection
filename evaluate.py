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

# evaluate.py

import os
import json
import cv2
import numpy as np
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ——— Test Dataset ———
class TestDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_files  = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        )
        self.img_dir   = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fn = self.img_files[idx]
        # 1) 读取并归一化图像
        img_path = os.path.join(self.img_dir, fn)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        img_t = torch.tensor(resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        # 2) 读取调整后的关键点（224×224 像素坐标）
        lbl_path = os.path.join(self.label_dir, fn.rsplit('.', 1)[0] + '.json')
        with open(lbl_path, 'r') as f:
            pts = np.array([
                [p['x'], p['y']] for p in json.load(f)['landmarks']
            ], dtype=np.float32)
        # 归一化到 [0,1]
        pts_norm = torch.from_numpy(pts.flatten() / 224.0).float()

        return fn, img_t, pts_norm

def evaluate():
    # —— 填入你的路径 —— 
    test_img_dir  = '/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedTestImg'
    test_lbl_dir  = '/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedTestLabel'
    model_weights = 'best_resnet18_dogkpt.pth'
    visuals_dir   = 'test_visuals'
    os.makedirs(visuals_dir, exist_ok=True)

    # —— 加载模型 —— 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 92)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device).eval()

    # —— 构建 DataLoader —— 
    ds = TestDataset(test_img_dir, test_lbl_dir)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0    # set to 0 on macOS to avoid multiprocessing issues
    )

    sum_nme, sum_rmse = 0.0, 0.0
    n = len(loader)

    with torch.no_grad():
        for (fn,), img_t, true_t in tqdm(loader, desc='Testing'):
            # 扩 batch
            img_batch = img_t.to(device)  # 直接用 batch 中的数据
            true_batch = true_t.to(device)             # [1,92]

            pred_flat = model(img_batch)[0].cpu().numpy().reshape(-1,2)
            true_flat = true_batch.cpu().numpy().reshape(-1,2)

            # 转回像素坐标
            pred_px = np.clip(pred_flat * 224.0, 0, 223)
            true_px = np.clip(true_flat * 224.0, 0, 223)

            dists = np.linalg.norm(pred_px - true_px, axis=1)
            sum_nme  += np.mean(dists) / 224.0
            sum_rmse += np.sqrt(np.mean(dists**2))

            # 可视化前5张
            if len(os.listdir(visuals_dir)) < 5:
                vis = img_batch[0].cpu().permute(1, 2, 0).numpy()
                vis = np.clip(vis * 255.0, 0, 255).astype(np.uint8).copy()

                for x,y in true_px.astype(int):
                    cv2.circle(vis, (x,y), 2, (0,255,0), -1)
                for x,y in pred_px.astype(int):
                    cv2.circle(vis, (x,y), 2, (0,0,255), -1)
                cv2.imwrite(os.path.join(visuals_dir, fn), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # —— 输出整体指标 —— 
    print(f'\n=== Test Results ===')
    print(f'Average NME  : {sum_nme/n:.4f}')
    print(f'Average RMSE : {sum_rmse/n:.1f} px')
    print(f'Sample visuals saved in ./{visuals_dir}/')

if __name__ == '__main__':
    evaluate()
