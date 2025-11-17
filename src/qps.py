import sys
import os
import argparse
import csv

current_dir = os.getcwd()
if current_dir.endswith('src'):
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.infra as infra
from src.config import *
from src.spat import *
from src.stats import *
from src.request import *
# 明确的字段顺序，和你给出的 energy_stats()/latency_stats() 定义一致
ENERGY_FIELDS = [
    "qkv",
    "spat_score_context",
    "spat_similarity",
    "spat_softmax",
    "proj",
    "comm",
    "norm",
    "moe",
    "ffn",
    "scheduling",
    "promotion",
    "balance",
    "sum",
    "mem_energy",
    "compute_energy",
]

LATENCY_FIELDS = [
    "qkv",
    "spat_score_context",
    "spat_similarity",
    "spat_softmax",
    "proj",
    "comm",
    "norm",
    "moe",
    "ffn",
    "scheduling",
    "promotion",
    "balance",
    "sum",
]

def str_to_bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="PIM Simulation Runner")

    parser.add_argument("--model", type=str, default="llama4",
                        help="Model name (default: llama4)")

    parser.add_argument("--length", type=int, default=1024,
                        help="Base sequence length (default: 8192)")

    parser.add_argument("--request", type=int, default=64,
                        help="Number of requests (default: 64)")

    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup iteration (default: 128)")

    parser.add_argument("--hw", type=str, default="pim",
                        help="Hardware configuration: pim or gpu (default: pim)")

    parser.add_argument("--scheduling_interval", type=int, default=16,
                        help="Scheduling interval (default: 64)")

    parser.add_argument("--dynamic_enable", type=str_to_bool, default=True,
                        help="Dynamic enable: True or False (default: True)")

    parser.add_argument("--scheduling_enable", type=str_to_bool, default=True,
                        help="Scheduling enable: True or False (default: True)")
    parser.add_argument("--test_mode", type=str_to_bool, default=False,
                        help="Test mode: True or False (default: True)")
    parser.add_argument("--qps", type=float, default=4,
                        help="QPS (default: 4)")
    parser.add_argument("--activation_ratio", type=float, default=0.1,
                        help="Activation ratio (default: 0.1)")
    parser.add_argument("--max_iter", type=int, default=1024,
                        help="Max iteration (default: 1024)")
    parser.add_argument("--tbt_output_dir", type=str, default="../output/qps",
                        help="Output directory (default: ../output/qps/qps.csv)")
    parser.add_argument("--stats_output_dir", type=str, default="../output/stats",
                        help="Output directory (default: ../output/qps/qps.csv)")
    parser.add_argument("--sparse_enable", type=str_to_bool, default=True,)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed (default: None)")
    return parser.parse_args()


def write_stats_csv(output_dir, model_name, length, request, sparse_enable, scheduling_enable, qps, hw, energy_stats_dict, latency_stats_dict):
    """
    把所有 energy 和 latency 字段写到 CSV 中。
    文件名: result_{model}_{length}_{request}_{hw}.csv
    列: model,length,request,hw, energy_<field>..., latency_<field>...
    """
    csv_file = f"{output_dir}/result_{model_name}_sparse_{sparse_enable}_scheduling_{scheduling_enable}_qps_{qps}_length_{length}_request_{request}_{hw}.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["model", "sparse_enable", "scheduling_enable", "qps", "length", "request", "hw"]
            header += [f"energy_{k}" for k in ENERGY_FIELDS]
            header += [f"latency_{k}" for k in LATENCY_FIELDS]
            writer.writerow(header)

        row = [model_name, sparse_enable, scheduling_enable, qps, length, request, hw]
        # energy fields (如果某个key缺失，则写0)
        for k in ENERGY_FIELDS:
            row.append(energy_stats_dict.get(k, 0))
        # latency fields
        for k in LATENCY_FIELDS:
            row.append(latency_stats_dict.get(k, 0))

        writer.writerow(row)

    print(f"[CSV] 已写入结果到 {csv_file}")


def write_tbt_csv(output_dir, model_name, sparse_enable, scheduling_enable, qps, hardware_name, tbt_stats_list):
    """
    将 tbt_stats、qps、model、hardware_name 保存为 CSV
    文件名: tbt_stats_{model}_{qps}_{hardware_name}.csv
    列: model, qps, hardware_name, iteration, request_batch_size, iteration_latency, 
        average_utilization, normalized_variance, scheduling_remainder
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = f"{output_dir}/tbt_stats_{model_name}_sparse_{sparse_enable}_scheduling_{scheduling_enable}_qps_{qps}_hardware_{hardware_name}.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [
                "model", "sparse_enable", "scheduling_enable", "qps", "hardware_name", "iteration", 
                "request_batch_size", "iteration_latency", 
                "average_utilization", "normalized_variance", "scheduling_remainder"
            ]
            writer.writerow(header)

        # 写入每一行数据
        # tbt_stats_list 中每个元素是: (request_batch_size, iteration_latency, average_utilization, normalized_variance, scheduling_remainder)
        for iteration, (request_batch_size, iteration_latency, average_utilization, normalized_variance, scheduling_remainder) in enumerate(tbt_stats_list, start=1):
            row = [
                model_name, sparse_enable, scheduling_enable, qps, hardware_name, iteration,
                request_batch_size, iteration_latency,
                average_utilization, normalized_variance, scheduling_remainder
            ]
            writer.writerow(row)

    print(f"[CSV] 已写入 TBT 统计到 {csv_file} (共 {len(tbt_stats_list)} 行)")


def write_pim_stats_csv(output_dir, model_name, length, request, sparse_enable, scheduling_enable, qps, hardware_name, pim_stats_dict):
    """
    将 pim_stats 保存为 CSV
    文件名: pim_stats_{model}_sparse_{sparse_enable}_scheduling_{scheduling_enable}_qps_{qps}_length_{length}_request_{request}_hardware_{hardware_name}.csv
    列: model, length, request, sparse_enable, scheduling_enable, qps, hardware_name, pim_latency, pim_energy, sparse_pim_latency, sparse_pim_energy
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = f"{output_dir}/pim_stats_{model_name}_sparse_{sparse_enable}_scheduling_{scheduling_enable}_qps_{qps}_length_{length}_request_{request}_hardware_{hardware_name}.csv"
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [
                "model", "request", "length", "sparse_enable", "scheduling_enable", "qps", "hardware_name",
                "pim_latency", "pim_energy", "sparse_pim_latency", "sparse_pim_energy"
            ]
            writer.writerow(header)

        row = [
            model_name, length, request, sparse_enable, scheduling_enable, qps, hardware_name,
            pim_stats_dict.get("pim_latency", 0),
            pim_stats_dict.get("pim_energy", 0),
            pim_stats_dict.get("sparse_pim_latency", 0),
            pim_stats_dict.get("sparse_pim_energy", 0)
        ]
        writer.writerow(row)
    
    print(f"[CSV] 已写入PIM统计到 {csv_file}")


if __name__ == "__main__":

    args = parse_args()

    print("========== Simulation Settings ==========")
    print(f"Model            : {args.model}")
    print(f"Sequence length  : {args.length}")
    print(f"Requests         : {args.request}")
    print(f"Warmup iteration : {args.warmup}")
    print(f"Hardware         : {args.hw}")
    print("==========================================\n")

    model = make_model_config(args.model)

    Hardware_config_PIM = {
        "device": DeviceType.HB_PIM,
        "n_device": 8,
        "GPU": make_gpu_config(8, hbf_en=True),
        "PIM": make_pim_config(num_pim_device=8, num_pim_stack=4, sparse_enable=args.sparse_enable),
    }

    Hardware_config_GPU = {
        "device": DeviceType.GPU,
        "n_device": 8,
        "GPU": make_gpu_config(8, hbf_en=False),
        "PIM": make_pim_config(num_pim_device=2, num_pim_stack=4, sparse_enable=args.sparse_enable),
    }

    ###new request stream

    tbt_stats = tbt_stats()
    pim_stats = pim_stats()

    if args.hw.lower() == "gpu" or args.hw.lower() == "ugpu":
        # 注意：这里使用命令行中的 model/length/request，这样和你传参一致
        request_tracking_table = None
        Request_Stream = Request_Stream(args.qps, args.activation_ratio, model, initial_request_count=args.request, request_tracking_table=request_tracking_table, dynamic_enable=args.dynamic_enable)
        Host_request_stream = Request_Stream
        Host_pim_profile_table = None
        Host_hbf_track_table = None

        # Host_request_stream.request_batch.gen_activated_clusters()
        # print(f"request_stream.request_batch.activated_clusters: {Host_request_stream.request_batch.activated_clusters}")
        gpu_energy_stats = energy_stats()
        gpu_latency_stats = latency_stats()

        gpu_system = infra.System(
            gpu_energy_stats,
            gpu_latency_stats,
            tbt_stats,
            model,
            Hardware_config_GPU,
            request_stream=Host_request_stream,
            pim_profile_table=Host_pim_profile_table,
            hbf_track_table=Host_hbf_track_table,
            warmup_iteration=0,
            scheduling_interval=args.scheduling_interval,
            scheduling_enable=args.scheduling_enable,
            dynamic_enable=args.dynamic_enable,
            test_mode=args.test_mode
        )
        gpu_system.system_setup()
        gpu_system.sim(max_iteration=args.max_iter)

        # 打印原有信息
        print(f"gpu_energy: {gpu_system.energy_stats}")
        print(f"-----------------------------")
        print(f"gpu_latency: {gpu_system.latency_stats}")
        print(f"-----------------------------")
        print(f"gpu_energy: {gpu_system.energy_stats.get('sum')}, gpu_latency: {gpu_system.latency_stats.get('sum')}")

        

        # 写 CSV：把 gpu_system.energy_stats（字典）和 gpu_system.latency_stats（字典）所有字段写入
        write_stats_csv(
            args.stats_output_dir,
            args.model,
            args.length,
            args.request,
            args.sparse_enable,
            args.scheduling_enable,
            args.qps,
            "gpu",
            gpu_system.energy_stats,
            gpu_system.latency_stats
        )
        
        # 写 TBT 统计 CSV
        write_tbt_csv(
            args.tbt_output_dir,
            args.model,
            args.sparse_enable,
            args.scheduling_enable,
            args.qps,
            "gpu",
            gpu_system.tbt_stats
        )

    else:
        request_tracking_table = {}
        PIM_request_stream = Request_Stream(args.qps, args.activation_ratio, model, initial_request_count=args.request, request_tracking_table=request_tracking_table, dynamic_enable=args.dynamic_enable)
        #PIM_request_stream = None
        PIM_pim_profile_table = PIM_Profile_Table(Hardware_config_PIM["PIM"])
        PIM_pim_profile_table.build_profile_table()
        
        # PIM_request_batch = Request_Batch(0.10, Model, args.seed)
        # for i in range(args.request):
        #     PIM_request_batch.append(i, args.length+256, args.length+256+256+4)
        for request_id, request_info in PIM_request_stream.request_batch.request.items():

            request_tracking_table[request_id] = HBF_Track_Table(1.5*request_info["total_cluster"] / 10)

        print(args.scheduling_enable)
        pim_energy_stats = energy_stats()
        pim_latency_stats = latency_stats()
        pim_stats = pim_stats

        pim_system = infra.System(
            pim_energy_stats,
            pim_latency_stats,
            tbt_stats,
            model,
            Hardware_config_PIM,
            request_batch=PIM_request_stream.request_batch,
            request_stream=PIM_request_stream,
            pim_profile_table=PIM_pim_profile_table,
            hbf_track_table=request_tracking_table,
            warmup_iteration=args.warmup,
            scheduling_interval=args.scheduling_interval,
            scheduling_enable=args.scheduling_enable,
            dynamic_enable=args.dynamic_enable,
            pim_stats=pim_stats,
            test_mode=args.test_mode
        )
        pim_system.system_setup()
        pim_system.sim(max_iteration=args.max_iter)

        # pim_system.system_setup()
        # pim_latency, pim_energy,sparse_pim_latency, sparse_pim_energy = pim_system.sim(max_iteration=args.max_iter)
        # pim_stats["sparse_pim_latency"] = sparse_pim_latency
        # pim_stats["sparse_pim_energy"] = sparse_pim_energy
        # pim_stats["pim_latency"] = pim_latency
        # pim_stats["pim_energy"] = pim_energy

        print("\n========== Simulation Finished ==========")
        print(f"pim_energy : {pim_system.energy_stats}")
        print("------------------------------------------")
        print(f"pim_latency: {pim_system.latency_stats}")
        print("------------------------------------------")
        print(f"SUMMARY → energy={pim_system.energy_stats.get('sum')}, latency={pim_system.latency_stats.get('sum')}")

        # 写 CSV：把 pim_system.energy_stats（字典）和 pim_system.latency_stats（字典）所有字段写入
        write_stats_csv(
            args.stats_output_dir,
            args.model,
            args.length,
            args.request,
            args.sparse_enable,
            args.scheduling_enable,
            args.qps,
            "pim",
            pim_system.energy_stats,
            pim_system.latency_stats
        )
        
        # 写 TBT 统计 CSV
        write_tbt_csv(
            args.tbt_output_dir,
            args.model,
            args.sparse_enable,
            args.scheduling_enable,
            args.qps,
            "pim",
            pim_system.tbt_stats
        )
        
        # 写 PIM 统计 CSV（如果存在pim_stats）
        if pim_system.pim_stats is not None:
            write_pim_stats_csv(
                "../output/sparse_stats",
                args.model,
                args.request,
                args.length,
                args.sparse_enable,
                args.scheduling_enable,
                args.qps,
                "pim",
                pim_system.pim_stats
            )