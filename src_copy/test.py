## Define models and layer.
## Generate models
import sys
import os
# 添加项目根目录到Python路径
# 如果notebook在src目录下，项目根目录是上一级目录
current_dir = os.getcwd()
if current_dir.endswith('src'):
    project_root = os.path.dirname(current_dir)
else:
    # 如果不在src目录，假设当前目录就是项目根目录
    project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import copy
import src.infra as infra
from src.config import *
from src.spat import *
from src.stats import *
Model = {
    "name":"llama4",
    "n_block":48,
    "n_experts":128,
    "experts_per_token":2,
    "n_q_head":40,
    "dim":5120,
    "hdim":16384,
    "hdim_moe":8192,
    "cluster_size":16,
    "n_kv_head":8,
    "dhead":128,
}


Hardware_config_GPU = {
    "device": DeviceType.GPU,
    "n_device": 8,
    "GPU": make_gpu_config(8, hbf_en=True),
    "PIM": make_pim_config(num_pim_device=2, num_pim_stack=4, sparse_enable=False),
}

Hardware_config_PIM = {
    "device": DeviceType.HB_PIM,
    "n_device": 2,
    "GPU": make_gpu_config(2, hbf_en=True),
    "PIM": make_pim_config(num_pim_device=2, num_pim_stack=4, sparse_enable=False),
}

Host_request_batch = infra.Request_Batch(0.05, Model)
for i in range(32):
    Host_request_batch.append(i, 8192+256, 8192+256+4)
Host_pim_profile_table = None
Host_hbf_track_table = None

# request_batch.gen_activated_clusters(0, 0)
# print(f"request_batch.activated_clusters: {request_batch.activated_clusters}")
gpu_energy_stats = energy_stats()
gpu_latency_stats = latency_stats()


gpu_system = infra.System(gpu_energy_stats, gpu_latency_stats, Model, Hardware_config_GPU, request_batch=Host_request_batch, request_stream=None, pim_profile_table=Host_pim_profile_table, \
    hbf_track_table=Host_hbf_track_table, warmup_iteration=0, scheduling_interval=4, scheduling_enable=False)
gpu_system.system_setup()
gpu_system.sim()
print(f"gpu_energy: {gpu_system.energy_stats}")
print(f"-----------------------------")
print(f"gpu_latency: {gpu_system.latency_stats}")
print(f"-----------------------------")

print(f"gpu_energy: {gpu_system.energy_stats['sum']}, gpu_latency: {gpu_system.latency_stats['sum']}")




# PIM_request_batch = infra.Request_Batch(0.05, Model)
# for i in range(32):
#     PIM_request_batch.append(i, 8192+256, 8192+256+64+4)
# PIM_pim_profile_table = PIM_Profile_Table(Hardware_config_PIM["PIM"])
# PIM_pim_profile_table.build_profile_table()
# PIM_hbf_track_table = HBF_Track_Table(65536553/4)
# pim_energy_stats = energy_stats()
# pim_latency_stats = latency_stats()
# pim_system = infra.System(pim_energy_stats, pim_latency_stats, Model, Hardware_config_PIM, request_batch=PIM_request_batch, request_stream=None, pim_profile_table=PIM_pim_profile_table, \
#         hbf_track_table=PIM_hbf_track_table, warmup_iteration=64, scheduling_interval=8)
# pim_system.system_setup()
# pim_system.sim(scheduling_enable=True)

# # print(f"gpu_energy: {gpu_system.energy_stats}, gpu_latency: {gpu_system.latency_stats}")
# # print(f"gpu_energy: {sum(gpu_system.energy_stats.values())}, gpu_latency: {sum(gpu_system.latency_stats.values())}")
# print(f"pim_energy: {pim_system.energy_stats}, pim_latency: {pim_system.latency_stats}")
# print(f"pim_energy: {sum(pim_system.energy_stats.values())}, pim_latency: {sum(pim_system.latency_stats.values())}")
