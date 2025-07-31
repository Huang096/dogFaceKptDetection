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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 配置
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------- HeatmapResNet 模型（与你训练时保持一致）--------
from train import HeatmapResNet  # 确保 train.py 中有 HeatmapResNet 定义

# --------- 数据集 ---------
class HeatmapTestDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img_path = os.path.join(self.img_dir, fn)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_t = self.transform(img)

        label_path = os.path.join(self.label_dir, fn.rsplit('.', 1)[0] + '.json')
        with open(label_path) as f:
            pts = np.array([[p['x'], p['y']] for p in json.load(f)['landmarks']], dtype=np.float32)
        # 将原始坐标按热图大小缩放
        pts[:, 0] *= (56 / 224)
        pts[:, 1] *= (56 / 224)

        return fn, img_t, pts, img  # img: 用于可视化

# --------- 后处理函数：heatmap → 坐标点 ---------
def heatmap_to_coord(hm):
    C, H, W = hm.shape
    coords = []
    for i in range(C):
        y, x = np.unravel_index(hm[i].argmax(), (H, W))
        coords.append([x, y])
    return np.array(coords, dtype=np.float32)

# --------- 主评估流程 ---------
def evaluate():
    test_img_dir = './data/adjustedTestImg'
    test_lbl_dir = './data/adjustedTestLabel'
    model_path   = './best_heatmap_resnet18.pth'
    vis_dir      = './test_visuals_heatmap'
    os.makedirs(vis_dir, exist_ok=True)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_ds = HeatmapTestDataset(test_img_dir, test_lbl_dir)
    num_pts = dummy_ds[0][2].shape[0]
    model = HeatmapResNet(num_pts=num_pts).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(dummy_ds, batch_size=1, shuffle=False)
    total_loss = 0.0

    for fn, img_t, true_pts, vis_img in tqdm(loader, desc="Eval"):
        fname = fn[0] if isinstance(fn, (list, tuple)) else fn
        img_t = img_t.to(device)
        with torch.no_grad():
            pred_hm = model(img_t)[0].cpu().numpy()  # [C,56,56]

        pred_pts = heatmap_to_coord(pred_hm)
        true_pts = true_pts[0].numpy()
        loss = np.mean((pred_pts - true_pts) ** 2)
        total_loss += loss

        # 可视化前 5 张样本
        if len(os.listdir(vis_dir)) < 5:
            # 可能从 DataLoader 中拿到的是 Tensor，要转回 NumPy 数组
            vis = vis_img[0]
            if isinstance(vis, torch.Tensor):
                vis = vis.cpu().numpy()
                # 如果维度是 (H, W, C)，无需转置；若是 (C, H, W)，请根据情况转置
            # 确保和坐标对齐
            vis = cv2.resize(vis, (224, 224))

            # 画出真实点（绿色）和预测点（红色）
            for x, y in true_pts.astype(int):
                cv2.circle(vis, (x*4, y*4), 2, (0, 255, 0), -1)
            for x, y in pred_pts.astype(int):
                cv2.circle(vis, (x*4, y*4), 2, (0, 0, 255), -1)

            out_path = os.path.join(vis_dir, fname)
            cv2.imwrite(out_path, vis)

    avg_rmse = np.sqrt(total_loss / len(loader))
    print(f"\n✅ Eval done. Average RMSE: {avg_rmse:.2f} px")
    print(f"Sample predictions saved to: {vis_dir}")

if __name__ == "__main__":
    evaluate()
