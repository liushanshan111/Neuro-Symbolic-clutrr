# utils/data_augmentation.py
"""
数据增强策略
"""
import torch
import random
import copy
from collections import defaultdict


class CLUTRRAugmenter:
    """CLUTRR数据增强器"""
    
    def __init__(self, 
                 oversample_hops=[3, 4, 5],
                 oversample_factor=3,
                 add_noise=True,
                 noise_prob=0.1):
        
        self.oversample_hops = oversample_hops
        self.oversample_factor = oversample_factor
        self.add_noise = add_noise
        self.noise_prob = noise_prob
    
    def augment_dataset(self, dataset):
        """
        增强数据集
        
        策略：
          1. 过采样长链样本
          2. 添加随机噪声（可选）
        """
        
        augmented_samples = []
        
        # 按跳数分组
        by_hop = defaultdict(list)
        for sample in dataset.samples:
            hop = sample['chain_length']
            by_hop[hop].append(sample)
        
        print(f"\n数据增强:")
        print(f"  原始样本: {len(dataset.samples)}")
        
        # 对每个跳数处理
        for hop, samples in by_hop.items():
            
            if hop in self.oversample_hops:
                # 过采样
                factor = self.oversample_factor
                print(f"  {hop}-hop: {len(samples)} → {len(samples) * factor} (×{factor})")
                
                for sample in samples:
                    # 原始样本
                    augmented_samples.append(sample)
                    
                    # 复制多份
                    for _ in range(factor - 1):
                        aug_sample = copy.deepcopy(sample)
                        
                        # 添加噪声（可选）
                        if self.add_noise and random.random() < self.noise_prob:
                            aug_sample = self._add_noise(aug_sample)
                        
                        augmented_samples.append(aug_sample)
            else:
                # 保持原样
                augmented_samples.extend(samples)
        
        print(f"  增强后总样本: {len(augmented_samples)}")
        
        # 更新数据集
        dataset.samples = augmented_samples
        
        return dataset
    
    def _add_noise(self, sample):
        """添加噪声（轻微扰动）"""
        # 这里可以添加各种噪声策略
        # 例如：替换一个关系为相似关系
        # 但要小心不要改变语义
        
        # 简单实现：不添加噪声（保持正确性）
        return sample


def create_balanced_loader(dataset, batch_size=32):
    """
    创建平衡的数据加载器
    
    确保每个batch中有不同跳数的样本
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # 计算每个样本的权重
    hop_counts = defaultdict(int)
    for sample in dataset.samples:
        hop_counts[sample['chain_length']] += 1
    
    # 反频率加权
    weights = []
    for sample in dataset.samples:
        hop = sample['chain_length']
        weight = 1.0 / hop_counts[hop]
        weights.append(weight)
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )
    
    # 创建加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    
    return loader