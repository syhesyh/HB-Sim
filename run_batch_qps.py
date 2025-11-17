#!/usr/bin/env python3
"""
多进程批量执行 qps.py 脚本
支持不同的硬件类型、模型和 QPS 参数组合
每个进程使用独立的随机数生成器状态
"""
import subprocess
import multiprocessing
import os
import sys
import time
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 参数组合
HARDWARE_TYPES = ["pim"]
MODELS = ["llama4", "qwen3"]
Scheduling_enable = [ "True"]
Scheduling_interval = [16]
Length = [1024]
Request = [256]
QPS_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
#QPS_VALUES = [0]
Seeds= np.random.randint(0, 2**31 - 1, size=1000)
# 默认参数
DEFAULT_ARGS = {
    "warmup": 0,
    "dynamic_enable": True,
    "max_iter": 512,
    "activation_ratio": 0.1,
    "tbt_output_dir": "./output/qps",
    "stats_output_dir": "./output/stats",
    "log_dir": "./logs",
}

# 进程锁，用于打印输出
print_lock = multiprocessing.Lock()


def run_qps_simulation(hw, model, qps, scheduling_enable, scheduling_interval, length, request, np_seed=None, randomstate=None, **kwargs):
    """
    运行QPS仿真
    
    Args:
        hw, model, qps, scheduling_enable, scheduling_interval, length, request: 仿真参数
        seed: 随机种子
        randomstate: numpy RandomState 对象，用于生成随机种子
        **kwargs: 其他参数
    """

    # 使用randomstate生成随机种子（如果提供）
    if randomstate is not None:
        seed = randomstate.randint(0, 2**31 - 1)
    else:
        seed = None
    
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
        "--seed", str(np_seed),
    ]
    
    
    task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}_scheduling_interval_{scheduling_interval}_length_{length}_request_{request}"
    
    # 创建日志目录
    log_dir = kwargs.get("log_dir", DEFAULT_ARGS["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(log_dir, f"{task_name}.log")
    
    # 准备环境变量（包含随机种子）
    env = os.environ.copy()
    if seed is not None:
        env['RANDOM_SEED'] = str(seed)
    
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行: {task_name}")
        if seed is not None:
            print(f"  随机种子: {seed}")
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
                env=env,  # 传递环境变量（包含随机种子）
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
                        error_msg = lines[-30:]  # 最后10行
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
    i = 0
    for hw in HARDWARE_TYPES:
        for model in MODELS:
            for qps in QPS_VALUES:
                for length in Length:
                    for request in Request:
                        for scheduling_enable in Scheduling_enable:
                            for scheduling_interval in Scheduling_interval:
                                # if scheduling_enable == "True" and scheduling_interval == 64:
                                #    tasks.append((hw, model, qps, scheduling_enable, scheduling_interval, length, request, Seeds[i]))
                                # if scheduling_enable == "False" and scheduling_interval == 4:
                                #     tasks.append((hw, model, qps, scheduling_enable, scheduling_interval, length, request, Seeds[i]))
                                tasks.append((hw, model, qps, scheduling_enable, scheduling_interval, length, request, Seeds[i]))
                                i += 1

    
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
    
    # 使用多进程执行
    max_workers = total_tasks 
    print(f"并发进程数: {max_workers}\n")
    
    # 使用 Manager 创建共享变量
    manager = multiprocessing.Manager()
    completed_count = manager.Value('i', 0)
    failed_list = manager.list()
    completed_lock = multiprocessing.Lock()
    
    # 进程执行函数
    def process_task(task_info, index, randomstate, np_seed, completed_count, completed_lock, failed_list, main_log_file, total_tasks):
        """在进程中执行任务"""
        hw, model, qps, scheduling_enable, scheduling_interval, length, request, seed = task_info
        
        # 每个进程启动前延迟1秒（第一个任务不延迟）
        if index > 0:
            time.sleep(1)
        
        try:
            success, task_name, error = run_qps_simulation(
                hw, model, qps, scheduling_enable, scheduling_interval, length, request,
                randomstate=randomstate,
                np_seed=np_seed
            )
            
            with completed_lock:
                completed_count.value += 1
                local_completed = completed_count.value
            
            # 写入主日志
            status = "✓ 成功" if success else "✗ 失败"
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {task_name}\n"
            if error:
                log_entry += f"  错误: {error[:2000]}\n"
            
            # 使用文件锁写入主日志
            with open(main_log_file, 'a', encoding='utf-8') as main_log:
                main_log.write(log_entry)
                main_log.flush()
            
            if not success:
                failed_list.append((task_name, error))
            
            # 显示进度
            with print_lock:
                print(f"进度: {local_completed}/{total_tasks} ({local_completed*100//total_tasks}%)")
                
        except Exception as e:
            with completed_lock:
                completed_count.value += 1
                local_completed = completed_count.value
            
            task_name = f"{hw}_{model}_qps{qps}_scheduling_{scheduling_enable}_scheduling_interval_{scheduling_interval}_length_{length}_request_{request}"
            failed_list.append((task_name, str(e)))
            
            with open(main_log_file, 'a', encoding='utf-8') as main_log:
                main_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 异常: {task_name} - {str(e)}\n")
                main_log.flush()
            
            with print_lock:
                print(f"进度: {local_completed}/{total_tasks} ({local_completed*100//total_tasks}%)")
                print(f"✗ 异常: {task_name} - {str(e)}")
    
    # 打开主日志文件
    with open(main_log_file, 'w', encoding='utf-8') as main_log:
        main_log.write(f"========== 批量执行 QPS 仿真 ==========\n")
        main_log.write(f"硬件类型: {HARDWARE_TYPES}\n")
        main_log.write(f"模型: {MODELS}\n")
        main_log.write(f"QPS 值: {QPS_VALUES}\n")
        main_log.write(f"总任务数: {total_tasks}\n")
        main_log.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        main_log.write("=" * 50 + "\n\n")
    
    # 创建主随机状态生成器，用于为每个进程生成独立的随机状态
    main_randomstate = np.random.RandomState()
    
    # 创建进程列表，每个进程使用独立的随机数生成器状态
    procs = [
        multiprocessing.Process(
            target=process_task,
            args=(task, i, np.random.RandomState(main_randomstate.randint(0, 2**31)), Seeds[i], completed_count, completed_lock, failed_list, main_log_file, total_tasks)
        )
        for i, task in enumerate(tasks)
    ]
    
    # 启动所有进程
    for proc in procs:
        proc.start()
    
    # 等待所有进程完成
    for proc in procs:
        proc.join()
    
    # 从共享列表中获取失败任务
    failed = list(failed_list)
    
    # 写入总结
    with open(main_log_file, 'a', encoding='utf-8') as main_log:
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

