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
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

# ---- 配置 ImageNet 归一化参数 ----
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# ---- 生成单通道 Gaussian Heatmap ----
def make_gaussian_heatmap(h, w, cx, cy, sigma=2):
    xs = torch.arange(0, w, dtype=torch.float32)
    ys = torch.arange(0, h, dtype=torch.float32)[:, None]
    g = torch.exp(-((xs[None, :] - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
    return g

# --- Heatmap 数据集 ---
class HeatmapDataset(Dataset):
    def __init__(self, img_dir, label_dir,
                 heatmap_size=(56,56), sigma=2,
                 transform=None):
        self.img_dir     = img_dir
        self.label_dir   = label_dir
        self.files       = sorted(f for f in os.listdir(img_dir)
                                  if f.lower().endswith(('.jpg','.png')))
        self.hm_h, self.hm_w = heatmap_size
        self.sigma       = sigma
        self.transform   = transform or transforms.ToTensor()
        # 读取第一个文件获取关键点数量
        sample_json = os.path.join(label_dir,
            self.files[0].rsplit('.',1)[0] + '.json')
        with open(sample_json,'r') as f:
            self.num_pts = len(json.load(f)['landmarks'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        # 1) 读取 & 随机几何增强
        img = cv2.imread(os.path.join(self.img_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 随机旋转±30° & 缩放[0.8,1.2]
        angle = random.uniform(-30, 30)
        scale = random.uniform(0.8, 1.2)
        h0, w0 = img.shape[:2]
        M = cv2.getRotationMatrix2D((w0/2, h0/2), angle, scale)
        img = cv2.warpAffine(
            img, M, (w0, h0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        img = cv2.resize(img, (224,224))

        # 2) ToTensor + ImageNet 归一化
        img_t = self.transform(img)
        img_t = (img_t - IMAGENET_MEAN[:,None,None]) / IMAGENET_STD[:,None,None]

        # 3) 读取并仿射变换标签
        with open(os.path.join(
            self.label_dir, fn.rsplit('.',1)[0] + '.json'
        )) as f:
            pts = np.array([
                [p['x'], p['y']]
                for p in json.load(f)['landmarks']
            ], dtype=np.float32)
        ones    = np.ones((pts.shape[0],1), dtype=np.float32)
        pts_aff = (M @ np.hstack([pts, ones]).T).T  # [N,2]
        # 缩放到 heatmap 大小
        pts_aff[:,0] *= (self.hm_w / 224)
        pts_aff[:,1] *= (self.hm_h / 224)

        # ——— 新增：清洗 & clamp 坐标 ———
        # 把 NaN/Inf 先替换成 -1
        pts_aff = np.nan_to_num(pts_aff,
                                nan=-1.0,
                                posinf=-1.0,
                                neginf=-1.0)
        # 再 clamp 到 [0, W-1] / [0, H-1]
        pts_aff[:,0] = np.clip(pts_aff[:,0], 0, self.hm_w - 1)
        pts_aff[:,1] = np.clip(pts_aff[:,1], 0, self.hm_h - 1)

        # 4) 构建 heatmaps
        hms = []
        for x, y in pts_aff:
            hm = make_gaussian_heatmap(
                self.hm_h, self.hm_w,
                cx=x, cy=y,
                sigma=self.sigma
            )
            hms.append(hm)
        heatmaps = torch.stack(hms, dim=0)  # [num_pts, H, W]

        return img_t, heatmaps


# --- Heatmap 回归模型：ResNet18 + Deconv Head ---
class HeatmapResNet(nn.Module):
    def __init__(self, num_pts, pretrained=True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # 去掉 avgpool + fc
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # [B,512,7,7]
        # 上采样 head -> 56×56
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_pts, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.deconv(x)
        return x  # [B, num_pts, 56, 56]

# --- 主训练流程 ---
def run_heatmap():
    # 超参数
    img_dir    = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedImg"
    label_dir  = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedLabel"
    batch_size = 16
    epochs     = 50
    lr         = 1e-3

    best_val   = float('inf')

    # 数据 & 划分
    ds = HeatmapDataset(
        img_dir, label_dir,
        heatmap_size=(56,56),
        sigma=2
    )
    n = len(ds)
    n_tr = int(n * 0.8)
    tr_ds, val_ds = random_split(
        ds, [n_tr, n-n_tr],
        generator=torch.Generator().manual_seed(42)
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader= DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HeatmapResNet(num_pts=ds.num_pts, pretrained=True).to(device)

    # 损失 & 优化 & 调度
    crit  = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(),
    lr=5e-4,    # ↓ 降低 lr
    weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,
                                                       T_max=epochs)

    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0

        for imgs, hms in tqdm(tr_loader, desc=f"[{ep}/{epochs}] Train"):
            imgs, hms = imgs.to(device), hms.to(device)

            # 前向 + 计算 loss
            pred = model(imgs)
            loss = crit(pred, hms)

            # 数值稳定 debug（可选，遇到 nan 就停）
            if torch.isnan(loss):
                print("Encountered NaN loss on epoch", ep)
                print("pred:", pred.mean().item(), pred.std().item(),
                      "hms:", hms.mean().item(), hms.std().item())
                raise ValueError("NaN loss")

            optim.zero_grad()
            loss.backward()

            # 梯度裁剪 ↓
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            tot_loss += loss.item()

        avg_tr = tot_loss / len(tr_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_loader:
                imgs, hms = imgs.to(device), hms.to(device)
                pred = model(imgs)
                val_loss += crit(pred, hms).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {ep}: train_loss={avg_tr:.4f}, val_loss={avg_val:.4f}")

        # 保存最优
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_heatmap_resnet18.pth")

        # 每 epoch 调度一次 ↓
        sched.step()

    print("✅ Finished. Best val_loss:", best_val)

if __name__ == "__main__":
    run_heatmap()
