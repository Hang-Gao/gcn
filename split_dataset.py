#!/usr/bin/env python3
"""
数据集分割程序
将alpha_perf_OS.csv中的539个样本随机分割为：
- 500个样本的训练集 (alpha_train_500.csv)
- 39个样本的测试集 (alpha_test_39.csv)
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

def split_dataset(input_file='alpha_perf_OS.csv', train_size=500, random_seed=42):
    """
    分割数据集
    
    Args:
        input_file: 输入CSV文件路径
        train_size: 训练集样本数量
        random_seed: 随机种子，确保结果可重现
    """
    print(f"开始读取数据文件: {input_file}")
    
    # 检查文件是否存在
    if not Path(input_file).exists():
        raise FileNotFoundError(f"文件 {input_file} 不存在")
    
    # 读取数据
    df = pd.read_csv(input_file)
    total_samples = len(df)
    test_size = total_samples - train_size
    
    print(f"原始数据集: {total_samples} 个样本")
    print(f"计划分割: 训练集 {train_size} 个，测试集 {test_size} 个")
    
    if train_size >= total_samples:
        raise ValueError(f"训练集大小 ({train_size}) 不能大于等于总样本数 ({total_samples})")
    
    # 设置随机种子
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 创建索引并随机打乱
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # 分割索引
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 根据索引分割数据
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # 保存分割后的数据
    train_file = 'alpha_train_500.csv'
    test_file = 'alpha_test_39.csv'
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"✅ 训练集保存至: {train_file} ({len(train_df)} 个样本)")
    print(f"✅ 测试集保存至: {test_file} ({len(test_df)} 个样本)")
    
    # 验证分割结果
    verify_split(train_df, test_df, total_samples)
    
    return train_file, test_file

def verify_split(train_df, test_df, original_total):
    """验证分割结果"""
    print("\n=== 分割结果验证 ===")
    
    train_count = len(train_df)
    test_count = len(test_df)
    total_after_split = train_count + test_count
    
    print(f"训练集样本数: {train_count}")
    print(f"测试集样本数: {test_count}")
    print(f"分割后总数: {total_after_split}")
    print(f"原始总数: {original_total}")
    
    # 检查总数是否正确
    if total_after_split == original_total:
        print("✅ 样本总数验证通过")
    else:
        print("❌ 样本总数验证失败")
        return False
    
    # 检查是否有重叠（通过比较第一列的值）
    train_first_col = set(train_df.iloc[:, 0].values)
    test_first_col = set(test_df.iloc[:, 0].values)
    overlap = train_first_col.intersection(test_first_col)
    
    if len(overlap) == 0:
        print("✅ 训练集和测试集无重叠")
        return True
    else:
        print(f"❌ 训练集和测试集有 {len(overlap)} 个重叠样本")
        return False

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("Alpha表达式数据集分割程序")
        print("=" * 50)
        
        # 执行分割
        train_file, test_file = split_dataset()
        
        print("\n" + "=" * 50)
        print("分割完成!")
        print(f"新训练集: {train_file}")
        print(f"新测试集: {test_file}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")