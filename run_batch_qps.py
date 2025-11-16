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
HARDWARE_TYPES = ["pim"]
MODELS = ["llama4", "qwen"]
Scheduling_enable = ["False", "True"]
Scheduling_interval = [4,64]
Length = [1024, 2048, 4096]
Request = [64, 128, 256]
#QPS_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
QPS_VALUES = [0]
# 默认参数
DEFAULT_ARGS = {
    "warmup": 0,
    "dynamic_enable": True,
    "max_iter": 256,
    "activation_ratio": 0.1,
    "tbt_output_dir": "./output/qps",
    "stats_output_dir": "./output/stats",
    "log_dir": "./logs",
}

# 线程锁，用于打印输出
print_lock = threading.Lock()


def run_qps_simulation(hw, model, qps, scheduling_enable, scheduling_interval, length, request, **kwargs):

    cmd = [
        sys.executable,
        "src/qps.py",
        "--hw", str(hw),
        "--model", str(model),
        "--qps", str(qps),
        "--length", str(length),
        "--request", str(request),
        "--warmup", str(kwargs.get("warmup", DEFAULT_ARGS["warmup"])),
        "--scheduling_interval", str(scheduling_interval),
        "--max_iter", str(kwargs.get("max_iter", DEFAULT_ARGS["max_iter"])),
        "--activation_ratio", str(kwargs.get("activation_ratio", DEFAULT_ARGS["activation_ratio"])),
        "--scheduling_enable", str(scheduling_enable),
        "--tbt_output_dir", str(kwargs.get("tbt_output_dir", DEFAULT_ARGS["tbt_output_dir"])),
        "--stats_output_dir", str(kwargs.get("stats_output_dir", DEFAULT_ARGS["stats_output_dir"])),
    ]
    
    
    task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}"
    
    # 创建日志目录
    log_dir = kwargs.get("log_dir", DEFAULT_ARGS["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(log_dir, f"{task_name}.log")
    
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行: {task_name}")
        print(f"  日志文件: {log_file}")
    
    try:
        # 执行命令，将输出保存到日志文件
        with open(log_file, 'w', encoding='utf-8') as log_f:
            log_f.write(f"========== 任务开始: {task_name} ==========\n")
            log_f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"命令: {' '.join(cmd)}\n")
            log_f.write("=" * 60 + "\n\n")
            
            result = subprocess.run(
                cmd,
                cwd=current_dir,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # 将 stderr 也重定向到 stdout
                text=True,
            )
            
            log_f.write("\n" + "=" * 60 + "\n")
            log_f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"返回码: {result.returncode}\n")
            log_f.write(f"========== 任务结束: {task_name} ==========\n")
        
        if result.returncode == 0:
            with print_lock:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 完成: {task_name}")
            return True, task_name, None
        else:
            # 读取日志文件的最后几行作为错误信息
            error_msg = "Unknown error"
            try:
                with open(log_file, 'r', encoding='utf-8') as log_f:
                    lines = log_f.readlines()
                    if lines:
                        error_msg = lines[-10:]  # 最后10行
                        error_msg = '\n'.join(error_msg).strip()[:500]
            except:
                pass
            
            with print_lock:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 失败: {task_name}")
                print(f"  错误: {error_msg[:200]}")
            return False, task_name, error_msg
            
    except subprocess.TimeoutExpired:
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n任务超时: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        with print_lock:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 超时: {task_name}")
        return False, task_name, "Timeout"
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n异常: {str(e)}\n")
        with print_lock:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 异常: {task_name}")
            print(f"  异常: {str(e)}")
        return False, task_name, str(e)


def main():
    """主函数：生成所有参数组合并多线程执行"""
    # 创建日志目录
    log_dir = DEFAULT_ARGS.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成所有任务
    tasks = []
    for hw in HARDWARE_TYPES:
        for model in MODELS:
            for qps in QPS_VALUES:
                for length in Length:
                    for request in Request:
                        for scheduling_enable in Scheduling_enable:
                            for scheduling_interval in Scheduling_interval:
                                if scheduling_enable == "True" and scheduling_interval == 64:
                                   tasks.append((hw, model, qps, scheduling_enable, scheduling_interval, length, request))
                                if scheduling_enable == "False" and scheduling_interval == 4:
                                    tasks.append((hw, model, qps, scheduling_enable, scheduling_interval, length, request))
    
    total_tasks = len(tasks)
    
    # 创建主日志文件
    main_log_file = os.path.join(log_dir, f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    print(f"========== 批量执行 QPS 仿真 ==========")
    print(f"硬件类型: {HARDWARE_TYPES}")
    print(f"模型: {MODELS}")
    print(f"QPS 值: {QPS_VALUES}")
    print(f"总任务数: {total_tasks}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志目录: {log_dir}")
    print(f"主日志文件: {main_log_file}")
    print("=" * 50)
    
    # 使用线程池执行
    max_workers = total_tasks 
    print(f"并发线程数: {max_workers}\n")
    
    completed = 0
    failed = []
    
    # 打开主日志文件
    with open(main_log_file, 'w', encoding='utf-8') as main_log:
        main_log.write(f"========== 批量执行 QPS 仿真 ==========\n")
        main_log.write(f"硬件类型: {HARDWARE_TYPES}\n")
        main_log.write(f"模型: {MODELS}\n")
        main_log.write(f"QPS 值: {QPS_VALUES}\n")
        main_log.write(f"总任务数: {total_tasks}\n")
        main_log.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        main_log.write("=" * 50 + "\n\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(run_qps_simulation, hw, model, qps, scheduling_enable, scheduling_interval, length, request): (hw, model, qps, scheduling_enable, scheduling_interval, length, request)
                for hw, model, qps, scheduling_enable, scheduling_interval, length, request in tasks
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                hw, model, qps, scheduling_enable, scheduling_interval, length, request = future_to_task[future]
                try:
                    success, task_name, error = future.result()
                    completed += 1
                    
                    # 写入主日志
                    status = "✓ 成功" if success else "✗ 失败"
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {task_name}\n"
                    if error:
                        log_entry += f"  错误: {error[:200]}\n"
                    main_log.write(log_entry)
                    main_log.flush()
                    
                    if not success:
                        failed.append((task_name, error))
                    
                    # 显示进度
                    with print_lock:
                        print(f"进度: {completed}/{total_tasks} ({completed*100//total_tasks}%)")
                        
                except Exception as e:
                    completed += 1
                    task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}_scheduling_interval_{scheduling_interval}_length_{length}_request_{request}"
                    failed.append((task_name, str(e)))
                    main_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 异常: {task_name} - {str(e)}\n")
                    main_log.flush()
                    with print_lock:
                        print(f"进度: {completed}/{total_tasks} ({completed*100//total_tasks}%)")
                        print(f"✗ 异常: {task_name} - {str(e)}")
        
        # 写入总结
        main_log.write("\n" + "=" * 50 + "\n")
        main_log.write(f"========== 执行完成 ==========\n")
        main_log.write(f"总任务数: {total_tasks}\n")
        main_log.write(f"成功: {total_tasks - len(failed)}\n")
        main_log.write(f"失败: {len(failed)}\n")
        main_log.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if failed:
            main_log.write("\n失败的任务:\n")
            for task_name, error in failed:
                main_log.write(f"  - {task_name}: {error[:200]}\n")
    
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

