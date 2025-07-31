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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

# ---- 配置 ImageNet 归一化参数 ----
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# --- 数据集定义 ---
class DogLandmarkDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.files     = [f for f in os.listdir(img_dir)
                          if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))

        img = self.transform(img)
        img = (img - IMAGENET_MEAN[:,None,None]) / IMAGENET_STD[:,None,None]

        lab = json.load(open(os.path.join(
            self.label_dir, fn.rsplit('.',1)[0] + '.json')))
        pts = np.array([[p['x'], p['y']] for p in lab['landmarks']],
                       dtype=np.float32) / 224.0
        pts = np.nan_to_num(pts, nan=0.0, posinf=1.0, neginf=0.0)
        pts = np.clip(pts, 0.0, 1.0)
        pts = (pts * 2.0) - 1.0
        pts = torch.from_numpy(pts.flatten())  # [92]

        return img, pts

# --- 模型定义 ---
def get_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 92),
        nn.Tanh()
    )
    return m

# --- 主训练流程 ---
def run():
    # 超参数
    best_val        = 1e9
    epochs          = 50
    patience        = 15
    patience_counter= 0

    img_dir   = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedImg"
    label_dir = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedLabel"

    full_ds = DogLandmarkDataset(img_dir, label_dir)
    n = len(full_ds)
    n_val = int(n * 0.2)
    n_tr  = n - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model().to(device)
    crit   = nn.SmoothL1Loss()
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=5e-4,              # 下调峰值
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,            # 延长 warmup
        div_factor=10,
        final_div_factor=100
    )

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, labs in tqdm(train_loader, desc=f"[{ep}/{epochs}] Train"):
            imgs, labs = imgs.to(device), labs.to(device)
            preds = model(imgs)
            loss  = crit(preds, labs)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            total_loss += loss.item()
        avg_tr = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                preds = model(imgs)
                val_loss += crit(preds, labs).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {ep}: train_loss={avg_tr:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_resnet18_improved.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {ep}.")
                break

    print("✅ Finished. Best val_loss:", best_val)

if __name__ == "__main__":
    run()
