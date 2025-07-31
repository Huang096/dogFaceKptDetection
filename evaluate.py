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
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ——— ImageNet 归一化参数（要和训练时一致） ———
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

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
        # 1) 读取图像并 resize
        img_path = os.path.join(self.img_dir, fn)
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))

        # 2) ToTensor + ImageNet 标准化
        img_t = torch.from_numpy(resized.transpose(2,0,1)).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN[:,None,None]) / IMAGENET_STD[:,None,None]

        # 3) 读取标签 & 归一化到 [-1,1]
        lbl_path = os.path.join(self.label_dir, fn.rsplit('.',1)[0] + '.json')
        with open(lbl_path, 'r') as f:
            pts = np.array([
                [p['x'], p['y']] for p in json.load(f)['landmarks']
            ], dtype=np.float32)
        pts = pts / 224.0         # [0,1]
        pts = (pts * 2.0) - 1.0   # -> [-1,1]
        pts = torch.from_numpy(pts.flatten()).float()

        return fn, img_t, pts

def evaluate():
    test_img_dir  = '/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedTestImg'
    test_lbl_dir  = '/Users/huangzheheng/Desktop/dogFaceKptDetection/data/adjustedTestLabel'
    model_weights = '/Users/huangzheheng/Desktop/dogFaceKptDetection/best_resnet18_improved.pth'
    visuals_dir   = 'test_visuals'
    os.makedirs(visuals_dir, exist_ok=True)

    # —— 加载模型 —— 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 92),
        torch.nn.Tanh()
    )
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device).eval()

    # —— DataLoader —— 
    ds     = TestDataset(test_img_dir, test_lbl_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    sum_nme, sum_rmse = 0.0, 0.0
    n = len(loader)

    with torch.no_grad():
        for fn, img_t, true_t in tqdm(loader, desc='Testing'):
            img_t   = img_t.to(device)            # [1,3,224,224]
            true_t  = true_t.to(device)           # [1,92]
            pred_t  = model(img_t)                # [1,92]

            # 恢复到像素坐标 [0,223]
            pred_pts = pred_t.cpu().numpy().reshape(-1,2)
            pred_px  = np.clip((pred_pts + 1)/2 * 224.0, 0, 223)

            true_pts = true_t.cpu().numpy().reshape(-1,2)
            true_px  = np.clip((true_pts + 1)/2 * 224.0, 0, 223)

            # 计算 NME & RMSE
            dists = np.linalg.norm(pred_px - true_px, axis=1)
            sum_nme  += np.mean(dists) / 224.0
            sum_rmse += np.sqrt(np.mean(dists**2))

            # 保存前 5 张可视化
            if len(os.listdir(visuals_dir)) < 5:
                vis = (resized := img_t[0].cpu().permute(1,2,0).numpy())
                # 把标准化前的像素恢复并画点
                vis = np.clip((vis * IMAGENET_STD.numpy() + IMAGENET_MEAN.numpy())*255,0,255).astype(np.uint8)
                for x,y in true_px.astype(int):
                    cv2.circle(vis, (x,y), 2, (0,255,0), -1)
                for x,y in pred_px.astype(int):
                    cv2.circle(vis, (x,y), 2, (0,0,255), -1)
                cv2.imwrite(os.path.join(visuals_dir, fn[0]), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f'\n=== Test Results ===')
    print(f'Average NME  : {sum_nme/n:.4f}')
    print(f'Average RMSE : {sum_rmse/n:.1f} px')
    print(f'Sample visuals saved in ./{visuals_dir}/')

if __name__ == '__main__':
    evaluate()
