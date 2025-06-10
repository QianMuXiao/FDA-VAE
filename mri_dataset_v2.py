import os
from torch.utils.data import Dataset

import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage
import random



def resize_and_normalize_array(input_array, target_shape, order=1):
    # 检查输入数组和目标形状的维度是否匹配
    if len(input_array.shape) != len(target_shape):
        raise ValueError("输入数组的维度和目标形状的维度不匹配。")

    # 计算缩放因子
    scale_factors = [target_shape[i] / input_array.shape[i] for i in range(len(target_shape))]

    # 使用 scipy.ndimage.zoom 进行缩放
    resized_array = scipy.ndimage.zoom(input_array, scale_factors, order=order)

    # 归一化：使用 np.interp 将数组值映射到 [0, 1] 范围
    min_val = np.min(resized_array)
    max_val = np.max(resized_array)

    if max_val > min_val:
        normalized_resized_array = np.interp(resized_array, (min_val, max_val), (-1, 1))
    else:
        # 如果 max == min，直接返回全0数组
        normalized_resized_array = np.zeros_like(resized_array)
    # normalized_resized_array = np.interp(resized_array, (min_val, max_val), (0, 1))
    
    return normalized_resized_array

class MultiModalDataset(Dataset):
    def __init__(self, data_path, lesion_patient_file, mode='paired', split='train', val_ratio=0.2, image_size=(160, 160), random_seed=42, phase_1 = 'pre', phase_2 = 'delay'):
        """
        Args:
            data_path (str): 根数据路径，包含'delay'和'pre'两个子目录。
            lesion_patient_file (str): 包含病人 ID 和病灶分类的文件路径。
            mode (str): 'paired' 或 'unpaired'，用于控制数据加载模式。
            split (str): 'train' 或 'val'，用于控制是训练集还是验证集。
            val_ratio (float): 验证集比例，默认为 0.2。
            image_size (tuple, optional): 控制 resize 大小，如 (160, 160)。
            random_seed (int): 随机种子，默认为 42。
        """
        self.data_path = data_path
        self.mode = mode
        self.split = split
        self.image_size = image_size
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 读取病人 ID 和病灶分类
        self.lesion_patient_dict = self.load_lesion_patient_ids(lesion_patient_file)

        # 根据验证集比例划分病人 ID
        self.patient_ids = self.split_patient_ids(val_ratio)

        # 构建图像路径列表
        self.delay_images = []
        self.pre_images = []

        for p_id in self.patient_ids:
            pre_dir = os.path.join(data_path, self.phase_1)
            delay_dir = os.path.join(data_path, self.phase_2)
            

            # 获取该患者的所有 delay 和 pre 文件
            delay_files = [f for f in os.listdir(delay_dir) if f.startswith(p_id + '_') and f.endswith('.nii.gz')]
            pre_files = [f for f in os.listdir(pre_dir) if f.startswith(p_id + '_') and f.endswith('.nii.gz')]

            # 将文件路径添加到列表中
            self.delay_images.extend([os.path.join(delay_dir, f) for f in delay_files])
            self.pre_images.extend([os.path.join(pre_dir, f) for f in pre_files])

        # 在成对模式下，构建图像对列表
        if self.mode == 'paired':
            self.image_pairs = []

            # 构建 delay 图像的字典，键为 (patient_id, slice_idx)
            delay_dict = {}
            for path in self.delay_images:
                filename = os.path.basename(path)
                p_id, slice_idx = self.parse_filename(filename)
                delay_dict[(p_id, slice_idx)] = path

            # 构建 pre 图像的字典，键为 (patient_id, slice_idx)
            pre_dict = {}
            for path in self.pre_images:
                filename = os.path.basename(path)
                p_id, slice_idx = self.parse_filename(filename)
                pre_dict[(p_id, slice_idx)] = path

            # 找到共同的键，构建图像对
            common_keys = set(delay_dict.keys()).intersection(set(pre_dict.keys()))
            for key in common_keys:
                self.image_pairs.append((delay_dict[key], pre_dict[key]))

            # 打乱图像对列表，确保随机性
            random.shuffle(self.image_pairs)
        else:
            # 在非成对模式下，不需要构建图像对
            # 打乱 delay_images 和 pre_images 列表
            random.shuffle(self.delay_images)
            random.shuffle(self.pre_images)

    def load_lesion_patient_ids(self, file_path):
        """
        读取包含病人 ID 和病灶分类的文件，返回字典 {lesion_class: [patient_ids]}
        """
        lesion_patient_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_lesion_class = None
        for line in lines:
            line = line.strip()
            if line.startswith('病灶分类'):
                current_lesion_class = line.split()[1].replace(':', '')
                lesion_patient_dict[current_lesion_class] = []
            elif line.startswith('病人 ID:'):
                patient_id = line.split(':')[1].strip()
                lesion_patient_dict[current_lesion_class].append(patient_id)

        return lesion_patient_dict

    def split_patient_ids(self, val_ratio):
        """
        根据验证集比例划分病人 ID，返回对应模式的病人 ID 列表
        """
        selected_patient_ids = []

        for lesion_class, patient_ids in self.lesion_patient_dict.items():
            # patient_ids = list(set(patient_ids))  # 去重
            patient_ids = sorted(list(set(patient_ids)))
            total_patients = len(patient_ids)
            val_count = max(1, int(total_patients * val_ratio))  # 至少选择一个病人作为验证集

            random.shuffle(patient_ids)

            if self.split == 'train':
                selected_ids = patient_ids[val_count:]
            else:  # 'val'
                selected_ids = patient_ids[:val_count]

            selected_patient_ids.extend(selected_ids)

        return selected_patient_ids

    def parse_filename(self, filename):
        """
        从文件名中解析出患者 ID 和切片索引

        Args:
            filename (str): 文件名，例如 'MR-391135_4.nii.gz'

        Returns:
            tuple: (patient_id, slice_idx)
        """
        name = filename.replace('.nii.gz', '')
        parts = name.split('_')
        patient_id = parts[0]
        slice_idx = parts[1]
        return patient_id, slice_idx

    def __len__(self):
        if self.mode == 'paired':
            return len(self.image_pairs)
        else:
            return max(len(self.delay_images), len(self.pre_images))

    def __getitem__(self, idx):
        if self.mode == 'paired':
            # 获取 delay 和 pre 图像路径
            delay_image_path, pre_image_path = self.image_pairs[idx]

            # 载入图像
            delay_image = sitk.GetArrayFromImage(sitk.ReadImage(delay_image_path))
            pre_image = sitk.GetArrayFromImage(sitk.ReadImage(pre_image_path))
        else:
            # 非成对模式，随机选择 delay 和 pre 图像
            delay_idx = idx % len(self.delay_images)
            pre_idx = random.randint(0, len(self.pre_images) - 1)

            delay_image_path = self.delay_images[delay_idx]
            pre_image_path = self.pre_images[pre_idx]

            # 载入图像
            delay_image = sitk.GetArrayFromImage(sitk.ReadImage(delay_image_path))
            pre_image = sitk.GetArrayFromImage(sitk.ReadImage(pre_image_path))

        # 调整大小并归一化
        if self.image_size:
            delay_image = resize_and_normalize_array(delay_image, target_shape=self.image_size)
            pre_image = resize_and_normalize_array(pre_image, target_shape=self.image_size)

        delay_image = torch.tensor(delay_image, dtype=torch.float32).unsqueeze(0)
        pre_image = torch.tensor(pre_image, dtype=torch.float32).unsqueeze(0)

        return delay_image, pre_image, delay_image_path, pre_image_path

# 验证函数
# def validate_dataset(dataset, batch_size=1, num_workers=0):
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     print("开始验证数据集...")
#     pbar = tqdm(enumerate(dataloader), desc="验证进度", total=len(dataloader))

#     for _, (delay, pre) in pbar:
#         delay_images, pre_images = delay, pre

#         pbar.set_postfix({
#             'delay_shape': delay_images.shape,
#             'pre_shape': pre_images.shape
#         })


    

