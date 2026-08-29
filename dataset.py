import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):
        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')

        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()

        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []

        for line in lines_rgb:
            image = line.split()[0]
            label = line.strip().split()[1]
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]

        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

        # 校验图像数量
        assert len(self.images) == len(self.images_rgbd), "RGB与深度图数量不匹配！"

        # ✅ 修正：分别定义RGB和深度图的transform
        # RGB图像（3通道）的transform
        self.rgb_transform = transforms.Compose([
            transforms.Resize((270, 480)),
            transforms.CenterCrop((256, 256)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # 深度图像（1通道）的transform
        self.depth_transform = transforms.Compose([
            transforms.Resize((270, 480)),
            transforms.CenterCrop((256, 256)),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 单通道均值/方差
        ])

    def __getitem__(self, index):
        # RGB图像读取
        img_rgb = cv2.imread(self.images[index])
        # 深度图像：读取为单通道灰度图
        img_rgbd = cv2.imread(self.images_rgbd[index], cv2.IMREAD_GRAYSCALE)

        try:
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
            img_rgbd = Image.fromarray(img_rgbd)
        except:
            print("图片读取错误：", self.images[index])
            
        # ✅ 修正：分别应用不同的transform
        img_rgb = self.rgb_transform(img_rgb)
        img_rgbd = self.depth_transform(img_rgbd)

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], \
            self.total_fat[index], self.total_carb[index], self.total_protein[index], img_rgbd

    def __len__(self):
        return len(self.images)