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

import os, json, random
import cv2, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm

# --- 数据集 ---
class DogLandmarkDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        # 读图
        img = cv2.imread(os.path.join(self.img_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img = self.transform(img)  # [3,224,224], float in [0,1]

        # 读标签
        lab = json.load(open(os.path.join(self.label_dir, fn.rsplit('.',1)[0]+'.json')))
        pts = np.array([[p['x'],p['y']] for p in lab['landmarks']], dtype=np.float32)
        pts /= 224.0                # 归一化到 [0,1]
        pts = torch.from_numpy(pts.flatten())  # 92

        return img, pts

# --- 模型 ---
def get_model():
    m = models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, 92)
    return m

# --- 训练/验证流程 ---
def run():
    # 路径
    img_dir   = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedImg"
    label_dir = "/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedLabel"

    # 构造数据集 & 划分
    full_ds = DogLandmarkDataset(img_dir, label_dir)
    n = len(full_ds)
    n_val = int(n*0.2)
    n_tr  = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_tr, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

    # 准备模型、损失、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model().to(device)
    crit   = nn.MSELoss()
    optim  = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

    epochs = 30
    best_val = 1e9

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tot = 0.0
        loop = tqdm(train_loader, desc=f"[{ep}/{epochs}] train ")
        for imgs, labs in loop:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = model(imgs)
            loss  = crit(preds, labs)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_tr = tot/len(train_loader)

        # ---- val ----
        model.eval()
        vt = 0.0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                preds = model(imgs)
                vt   += crit(preds, labs).item()
        avg_val = vt/len(val_loader)

        print(f"Epoch {ep}: train_loss={avg_tr:.4f}   val_loss={avg_val:.4f}")

        # 保存最优
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "best_resnet18_dogkpt.pth")

    print("✅ Finished. Best val_loss:", best_val)

if __name__=="__main__":
    run()
