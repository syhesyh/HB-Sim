#!/usr/bin/env python3
"""
多线程批量执行 qps.py 脚本
支持不同的硬件类型、模型和 QPS 参数组合
"""
import subprocess
import threading
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 参数组合
HARDWARE_TYPES = ["gpu", "pim"]
MODELS = ["llama4", "qwen"]
Scheduling_enable = [True, False]
QPS_VALUES = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

# 默认参数
DEFAULT_ARGS = {
    "length": 1024,
    "request": 64,
    "warmup": 0,
    "scheduling_interval": 8,
    "dynamic_enable": True,
    "max_iter": 4,
    "activation_ratio": 0.1,
    "tbt_output_dir": "./output/qps",
    "stats_output_dir": "./output/stats",
}

# 线程锁，用于打印输出
print_lock = threading.Lock()


def run_qps_simulation(hw, model, qps, scheduling_enable, **kwargs):
    """
    执行单个 qps.py 仿真任务
    
    Args:
        hw: 硬件类型 (gpu/pim)
        model: 模型名称 (llama4/qwen)
        qps: QPS 值
        **kwargs: 其他参数
    """
    # 构建命令
    cmd = [
        sys.executable,
        "src/qps.py",
        "--hw", str(hw),
        "--model", str(model),
        "--qps", str(qps),
        "--length", str(kwargs.get("length", DEFAULT_ARGS["length"])),
        "--request", str(kwargs.get("request", DEFAULT_ARGS["request"])),
        "--warmup", str(kwargs.get("warmup", DEFAULT_ARGS["warmup"])),
        "--scheduling_interval", str(kwargs.get("scheduling_interval", DEFAULT_ARGS["scheduling_interval"])),
        "--max_iter", str(kwargs.get("max_iter", DEFAULT_ARGS["max_iter"])),
        "--activation_ratio", str(kwargs.get("activation_ratio", DEFAULT_ARGS["activation_ratio"])),
        "--scheduling_enable", str(scheduling_enable),
        "--tbt_output_dir", str(kwargs.get("tbt_output_dir", DEFAULT_ARGS["tbt_output_dir"])),
        "--stats_output_dir", str(kwargs.get("stats_output_dir", DEFAULT_ARGS["stats_output_dir"])),
    ]
    
    
    task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}"
    
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行: {task_name}")
    
    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            cwd=current_dir,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            with print_lock:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 完成: {task_name}")
            return True, task_name, None
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            with print_lock:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 失败: {task_name}")
                print(f"  错误: {error_msg}")
            return False, task_name, error_msg
            
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 超时: {task_name}")
        return False, task_name, "Timeout"
    except Exception as e:
        with print_lock:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 异常: {task_name}")
            print(f"  异常: {str(e)}")
        return False, task_name, str(e)


def main():
    """主函数：生成所有参数组合并多线程执行"""
    # 生成所有任务
    tasks = []
    for hw in HARDWARE_TYPES:
        for model in MODELS:
            for qps in QPS_VALUES:
                for scheduling_enable in Scheduling_enable:
                    tasks.append((hw, model, qps, scheduling_enable))
    
    total_tasks = len(tasks)
    print(f"========== 批量执行 QPS 仿真 ==========")
    print(f"硬件类型: {HARDWARE_TYPES}")
    print(f"模型: {MODELS}")
    print(f"QPS 值: {QPS_VALUES}")
    print(f"总任务数: {total_tasks}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 使用线程池执行
    max_workers = min(8, total_tasks)  # 最多8个并发线程
    print(f"并发线程数: {max_workers}\n")
    
    completed = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_qps_simulation, hw, model, qps, scheduling_enable): (hw, model, qps, scheduling_enable)
            for hw, model, qps, scheduling_enable in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            hw, model, qps, scheduling_enable = future_to_task[future]
            try:
                success, task_name, error = future.result()
                completed += 1
                if not success:
                    failed.append((task_name, error))
                
                # 显示进度
                with print_lock:
                    print(f"进度: {completed}/{total_tasks} ({completed*100//total_tasks}%)")
                    
            except Exception as e:
                completed += 1
                task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}"
                failed.append((task_name, str(e)))
                with print_lock:
                    print(f"进度: {completed}/{total_tasks} ({completed*100//total_tasks}%)")
                    print(f"✗ 异常: {task_name} - {str(e)}")
    
    # 打印总结
    print("\n" + "=" * 50)
    print(f"========== 执行完成 ==========")
    print(f"总任务数: {total_tasks}")
    print(f"成功: {total_tasks - len(failed)}")
    print(f"失败: {len(failed)}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed:
        print("\n失败的任务:")
        for task_name, error in failed:
            print(f"  - {task_name}: {error}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()

