from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import copy
import cv2
import pandas as pd
import glob

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()


###############################
# newly added
###############################

allow_degradation_dict = {
    'hazy': True,
    'low-light': True,
    'motion-blurry': True,  # 值可以用于存储其他配置参数
    # 'raindrop': True,
    'rainy': True,
    'shadowed': True,
    # 'snowy': True,
    # 'uncompleted': True,
    # ...添加其他允许的类型
}

# 定义函数提取退化类型
def extract_degradation_type(path):
    parts = path.split('/')
    if 'train' in parts:
        train_index = parts.index('train') 
    elif 'val' in parts:
        train_index = parts.index('val')  # 找到train的位置
    else:
        raise ValueError(f"Invalid path: {path}, must have 'train' or 'val' in path")
    return parts[train_index + 1]       # train下一级目录是退化类型

# used to train image-to-text mapper
class UnpairedLQHQDataset(Dataset):
    def __init__(self,
                 csv_path,  # <- 修改1：增加csv_path参数
                 tokenizer=None,
                 size=512,
                 interpolation="bicubic",
                 placeholder_token="*",
                 template="a photo of a {}",
                 max_sample=2000,):
        super(UnpairedLQHQDataset, self).__init__()

        # 修改2：从csv加载数据路径和标题
        self.df = pd.read_csv(csv_path, sep='\t')  # 假设csv用制表符分隔

        # 添加退化类型列
        self.df['degration_type'] = self.df['filepath'].apply(extract_degradation_type)

        # 筛选退化类型
        self.df = self.df[self.df['degration_type'].isin(allow_degradation_dict.keys())]

        # 按退化类型分组并取前2000条
        self.df = self.df.groupby('degration_type').head(max_sample).reset_index(drop=True)

        # 生成对应的GT路径
        self.df['GT_path'] = self.df['filepath'].str.replace('/LQ/', '/GT/')

        self.image_paths = self.df['filepath'].tolist()
        self.GT_paths = self.df['GT_path'].tolist() # 添加GT路径
        # Update paths based on new base directory
        self.image_paths = [p.replace("/workspace/datasets/SD_Rest", "/data/coding") for p in self.image_paths]
        self.GT_paths = [p.replace("/workspace/datasets/SD_Rest", "/data/coding") for p in self.GT_paths]
        self.titles = [x.split(":")[0].strip() for x in self.df['title']]  # 提取冒号前的描述

        self.title_templates = [x.split(":")[0].strip() + " {}" for x in self.df['title']]

        # 修改3：调整类成员变量
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.tokenizer = tokenizer
        self.size = size
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        
        # 废弃原来的路径收集逻辑
        # self.dataroot_list = dataroot_list
        # self.image_paths = []
        # for dataroot in self.dataroot_list:
        #     self.image_paths.extend(sorted(glob.glob(os.path.join(dataroot, "*"))))
        self.placeholder_token = placeholder_token
        self.template = template
        self.patch_size = size
        """
        self.tokenizer = tokenizer
        self.size = size

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        """

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]

        return torchvision.transforms.Compose(transform_list)

    def process(self, image):

        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)

        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):

        ###############################################################
        example = {}

        # 修改4：使用csv中的真实文本
        
        # raw_text = self.titles[i % self.num_images]  # 直接使用标题中的描述
        # example["text"] = raw_text
        """
        # 修改5：简化tokenize流程（不需要占位符处理）
        example["input_ids"] = self.tokenizer(
            raw_text,  # 直接使用原始文本
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        """
        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        # text = self.title_templates[i % self.num_images].format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)
        if self.tokenizer is not None:
            example["input_ids"] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids[0]

        # 修改6：保留原有图像处理逻辑
        self.current_path = self.image_paths[i % self.num_images]
        image_name = self.current_path.split('/')[-1].split(".")[0]

        self.current_path_gt = self.GT_paths[i % self.num_images]
        image_name_gt = self.current_path_gt.split('/')[-1].split(".")[0]

        example["degration_type"] = extract_degradation_type(self.current_path)

        try:
            image = Image.open(self.current_path)
        
            if not image.mode == "RGB":
                image = image.convert("RGB")


            image_gt = Image.open(self.current_path_gt)
        
            if not image_gt.mode == "RGB":
                image_gt = image_gt.convert("RGB")

            H, W = image.size
            # 定义通用预处理（排除 RandomCrop）
            process = []

            # 需要裁剪时
            if H >= self.patch_size and W >= self.patch_size:
                # 获取随机裁剪参数
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(self.patch_size, self.patch_size)
                )
                
                # 对两个图像应用相同的裁剪坐标
                process += transforms.Lambda(lambda x: transforms.functional.crop(x, i, j, h, w)),
                # process += transforms.Crop(image, i, j, h, w)
                
                # 统一缩放到最终尺寸
                # process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
            else:
                # 小图直接中心裁剪+缩放
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                process += torchvision.transforms.CenterCrop(self.size),
            
            process += torchvision.transforms.ToTensor(),
            process = torchvision.transforms.Compose(process)
            # 转换为 Tensor
            torch_image = process(image)
            torch_image_gt = process(image_gt)

            """process = []
            if H < self.patch_size or W < self.patch_size:
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                process += torchvision.transforms.CenterCrop(self.size),
            else:
                process += torchvision.transforms.RandomCrop(self.size),
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                
            process += torchvision.transforms.ToTensor(),
            process = torchvision.transforms.Compose(process)

            torch_image = process(image)
            torch_image_gt = process(image_gt)"""

            # example["pixel_values"] = torch_image
            example["pixel_values_vae"] = torchvision.transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            )(torch_image)

            example["pixel_values_normal"] = torch_image
            example["pixel_values"] = example["pixel_values_vae"]
            example["conditioning_pixel_values"] = example["pixel_values"]

            example["pixel_values_vae_gt"] = torchvision.transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            )(torch_image_gt)
            example["pixel_values_gt"] = example["pixel_values_vae_gt"]


            example["pixel_values_clip"] = torchvision.transforms.Compose( # the clip process input range should be [0, 1]
                [torchvision.transforms.Resize((224, 224), interpolation=self.interpolation),
                    torchvision.transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    )
                    ]
            )(torch_image)

            """
            # PIL Image
            H, W = image.size
            if H < self.patch_size or W < self.patch_size:
                croper = torchvision.transforms.CenterCrop(H if H < W else W)
                image = croper(image)
                image_np = np.array(image)
            else:
                image_np = np.array(image)
                rnd_h_H = random.randint(0, max(0, H - self.patch_size))
                rnd_w_H = random.randint(0, max(0, W - self.patch_size))
                image_np = image_np[rnd_w_H : rnd_w_H + self.patch_size, rnd_h_H: rnd_h_H + self.patch_size,:]

            image_np = uint2single(image_np)

            example["pixel_values"] = self.process(image_np)

            ref_image_tensor = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)

            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)
            """

            example["image_name"] = image_name

        except Exception as e:

            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))

            example["image_name"] = image_name

            print("Bad Image Path", self.current_path)

        return example