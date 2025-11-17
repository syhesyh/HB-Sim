#!/usr/bin/env python3
"""
脚本用于提取 sparse_stats 目录中多个CSV文件的数据并合并到一个CSV文件中。
只提取每个文件的最后一行数据，保留所有列，包括：model, length, request, scheduling_enable 以及其他所有列
（如 sparse_enable, qps, hardware_name, pim_latency, pim_energy, sparse_pim_latency, sparse_pim_energy 等）
"""

import pandas as pd
import glob
import os
from pathlib import Path

def extract_sparse_stats(input_dir, output_file):
    """
    从输入目录的所有CSV文件中提取数据并保存到输出文件
    只提取每个文件的最后一行数据，保留所有列，确保包含：model, length, request, scheduling_enable
    
    Args:
        input_dir: 包含CSV文件的目录路径
        output_file: 输出CSV文件路径
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"警告: 在目录 {input_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 提取每个文件的数据
    all_data = []
    required_columns = ["model", "request", "length", "scheduling_enable"]
    
    for csv_file in sorted(csv_files):
        print(f"处理文件: {os.path.basename(csv_file)}")
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查必要的列是否存在
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告: {csv_file} 缺少必要的列: {missing_columns}")
                continue
            
            # 检查文件是否为空
            if len(df) == 0:
                print(f"警告: {csv_file} 是空文件，跳过")
                continue
            
            # 只提取最后一行数据
            last_row = df.iloc[-1:].copy()  # 使用 iloc[-1:] 保留 DataFrame 格式
            print(f"  -> 提取最后一行（共 {len(df)} 行）")
            all_data.append(last_row)
            
        except Exception as e:
            print(f"错误: 处理文件 {csv_file} 时出错: {e}")
            continue
    
    if not all_data:
        print("错误: 未能提取任何数据")
        return
    
    # 合并所有数据
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # 重新排列列的顺序，将重要的列放在前面
    # 首先确定所有可能的列顺序
    priority_columns = ["model", "request", "length", "scheduling_enable", "scheduling_interval", "sparse_enable", "qps", "hardware_name"]
    other_columns = [col for col in df_combined.columns if col not in priority_columns]
    ordered_columns = [col for col in priority_columns if col in df_combined.columns] + sorted(other_columns)
    
    # 重新排列列
    df_combined = df_combined[ordered_columns]
    
    # 按 model, request, length, scheduling_enable 排序
    df_combined = df_combined.sort_values(['model', 'request', 'length', 'scheduling_enable'])
    
    # 确保输出目录存在
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存到CSV文件
    df_combined.to_csv(output_file, index=False)
    print(f"\n成功提取并合并 {len(df_combined)} 行数据")
    print(f"结果已保存到: {output_file}")
    
    # 显示摘要
    print(f"\n包含的列: {', '.join(df_combined.columns)}")
    print("\n数据预览:")
    print(df_combined.head(10).to_string(index=False))
    
    print(f"\n数据统计:")
    print(f"  - 总行数: {len(df_combined)}")
    print(f"  - 唯一 model 数: {df_combined['model'].nunique()}")
    print(f"  - 唯一 request 数: {df_combined['request'].nunique()}")
    print(f"  - 唯一 length 数: {df_combined['length'].nunique()}")
    print(f"  - 唯一 scheduling_enable 值: {sorted(df_combined['scheduling_enable'].unique())}")

if __name__ == '__main__':
    # 设置输入和输出路径
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'output' / 'sparse_stats'
    output_file = script_dir / 'output' / 'sparse_stats_summary.csv'
    
    # 执行提取（保留所有列）
    extract_sparse_stats(str(input_dir), str(output_file))

