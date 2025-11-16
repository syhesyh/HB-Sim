#!/usr/bin/env python3
"""
脚本用于提取多个CSV文件的统计信息并合并到一个CSV文件中。
提取每个文件的iteration_latency的p50, p90, p99百分位数和最大值，
以及对应的model, qps, hardware_name信息。
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def extract_percentiles_from_csv(file_path):
    """
    从CSV文件中提取iteration_latency的百分位数、最大值和元数据
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        dict: 包含model, qps, hardware_name, p50, p90, p99, max的字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['model', 'qps', 'hardware_name', 'iteration_latency']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {file_path} 缺少必要的列")
            return None
        
        # 从第一行数据提取元数据（所有行的这些值应该相同）
        model = df['model'].iloc[0]
        qps = df['qps'].iloc[0]
        hardware_name = df['hardware_name'].iloc[0]
        # 提取scheduling_enable（如果存在）
        scheduling_enable = df['scheduling_enable'].iloc[0] if 'scheduling_enable' in df.columns else None
        
        # 计算百分位数和最大值
        latencies = df['iteration_latency'].values
        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        p99 = np.percentile(latencies, 99)
        max_value = np.max(latencies)
        
        result = {
            'model': model,
            'qps': qps,
            'hardware_name': hardware_name,
            'p50': p50,
            'p90': p90,
            'p99': p99,
            'max': max_value
        }
        
        # 如果存在scheduling_enable，添加到结果中
        if scheduling_enable is not None:
            result['scheduling_enable'] = scheduling_enable
        
        return result
    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时出错: {e}")
        return None

def extract_all_stats(input_dir, output_file):
    """
    从输入目录的所有CSV文件中提取统计信息并保存到输出文件
    
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
    
    # 提取每个文件的信息
    results = []
    for csv_file in sorted(csv_files):
        print(f"处理文件: {os.path.basename(csv_file)}")
        stats = extract_percentiles_from_csv(csv_file)
        if stats:
            results.append(stats)
    
    if not results:
        print("错误: 未能提取任何统计信息")
        return
    
    # 创建DataFrame
    df_results = pd.DataFrame(results)
    
    # 使用pivot将不同的hardware_name和scheduling_enable转换为不同的列
    metrics = ['p50', 'p90', 'p99', 'max']
    
    # 检查是否有scheduling_enable列
    has_scheduling = 'scheduling_enable' in df_results.columns
    
    # 获取所有唯一的hardware_name和scheduling_enable，并排序以确保列顺序一致
    hardware_names = sorted(df_results['hardware_name'].unique())
    if has_scheduling:
        scheduling_values = sorted(df_results['scheduling_enable'].unique())
    else:
        scheduling_values = [None]
    
    # 创建pivot表
    pivot_tables = []
    for metric in metrics:
        if has_scheduling:
            # 如果有scheduling_enable，同时按hardware_name和scheduling_enable分组
            pivot_df = df_results.pivot_table(
                index=['model', 'qps'],
                columns=['hardware_name', 'scheduling_enable'],
                values=metric,
                aggfunc='first'  # 如果有重复值，取第一个
            )
            # 重命名列，格式：{hardware_name}_{scheduling_enable}_{metric}
            # pivot_df.columns是MultiIndex，需要转换为元组列表
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [f'{hw}_{sched}_{metric}' for hw, sched in pivot_df.columns]
            else:
                pivot_df.columns = [f'{col}_{metric}' for col in pivot_df.columns]
        else:
            # 如果没有scheduling_enable，只按hardware_name分组
            pivot_df = df_results.pivot_table(
                index=['model', 'qps'],
                columns='hardware_name',
                values=metric,
                aggfunc='first'
            )
            # 重命名列，格式：{hardware_name}_{metric}
            pivot_df.columns = [f'{col}_{metric}' for col in pivot_df.columns]
        pivot_tables.append(pivot_df)
    
    # 合并所有pivot表
    df_final = pd.concat(pivot_tables, axis=1)
    
    # 重置索引，使model和qps成为普通列
    df_final = df_final.reset_index()
    
    # 重新排列列的顺序：先按metric分组，再按hardware_name，最后按scheduling_enable
    # 列名格式：{hardware_name}_{scheduling_enable}_{metric} 或 {hardware_name}_{metric}
    # 顺序：gpu_True_p50, gpu_False_p50, pim_True_p50, pim_False_p50, ...
    new_column_order = ['model', 'qps']
    for metric in metrics:
        for hw in hardware_names:
            if has_scheduling:
                for sched in scheduling_values:
                    col_name = f'{hw}_{sched}_{metric}'
                    if col_name in df_final.columns:
                        new_column_order.append(col_name)
            else:
                col_name = f'{hw}_{metric}'
                if col_name in df_final.columns:
                    new_column_order.append(col_name)
    
    # 重新排列列的顺序
    df_final = df_final[new_column_order]
    
    # 按model, qps排序
    df_final = df_final.sort_values(['model', 'qps'])
    
    # 保存到CSV文件
    df_final.to_csv(output_file, index=False)
    print(f"\n成功提取 {len(results)} 个文件的统计信息")
    print(f"结果已保存到: {output_file}")
    
    # 显示摘要
    print("\n结果预览:")
    print(df_final.to_string(index=False))

if __name__ == '__main__':
    # 设置输入和输出路径
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'output' / 'qps'
    output_file = script_dir / 'output' / 'qps_stats_summary.csv'
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行提取
    extract_all_stats(str(input_dir), str(output_file))

