from src.type import *

SCALING_FACTOR = {}
SCALING_FACTOR['MAX_COMPUTE_UTIL'] = 0.8
SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = 0.85
SCALING_FACTOR['HBM2E/HBM'] = 0.655
# ENERGY_TABLE: pJ per byte
# Cache info: https://core.ac.uk/download/pdf/232142915.pdf
ENERGY_TABLE = {
    'GPU': {},
    'SparsePIM': {},
    'CPU': {},
    'PIM': {
        PIMType.BA: {},
        PIMType.BG: {},
        PIMType.BUFFER: {}
    }
}
Energy_unit = 1e-12
ENERGY_TABLE['GPU']['reg'] = 0.0675 * Energy_unit
#4-way cache, ref: https://arxiv.org/pdf/1509.02308v1.pdf
ENERGY_TABLE['GPU'][ 'l1'] = 0.16 * 8 * Energy_unit
ENERGY_TABLE['GPU']['l2'] = 0.3 * 8 * Energy_unit
ENERGY_TABLE['GPU']['alu'] = 0.32 * Energy_unit
# ENERGY_TABLE['GPU']['hbm'] = (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8 * SCALING_FACTOR['HBM2E/HBM']
ENERGY_TABLE['GPU']['hbm'] = 3.88 * 8 * Energy_unit
# ENERGY_TABLE['GPU']['dimm'] = 10 * 8 * SCALING_FACTOR['HBM2E/HBM']
ENERGY_TABLE['GPU']['dimm'] = 20.31 * 8 * Energy_unit
ENERGY_TABLE['GPU']['hbf'] = 9.04 * 8 * Energy_unit
# ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['GPU'][ 'comm'] = 1.3 * 8 * Energy_unit


## 2017 MICRO FGDRAM
## https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
## Cell (ACT/PRE) energy: 0.11pJ/b,
## Cell (RD/WRT) energy: 0.44pJ/b,

## RD/WR Energy (column decoder to BG MUX): 1.01 pJ/b
## RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/b
## TSV energy : 0.5 pJ/b
## Silicon interposer IO energy : 0.3 pJ/b

## energy_table = [energy between DRAM cell and PE, energy between PE and buffer die
ENERGY_TABLE['SparsePIM']['row'] = 909 * Energy_unit #pJ/1KB
ENERGY_TABLE['SparsePIM']['read'] = 0.44*256 * Energy_unit #pJ/32B
ENERGY_TABLE['SparsePIM']['compute'] = 0.32*16 * Energy_unit #pJ/16MAC



# ENERGY_TABLE['PIM'][PIMType.BA]['mem'] = (0.11 +
#                                           0.44) * 8 * SCALING_FACTOR['HBM2E/HBM'] #, (1.01 + 1.23 + 0.5) * 8]
# ENERGY_TABLE['PIM'][PIMType.BG]['mem'] = (0.11 + 0.44 +
#                                           1.01) * 8 * SCALING_FACTOR['HBM2E/HBM'] #, (1.23 + 0.5) * 8]
# ENERGY_TABLE['PIM'][PIMType.BUFFER]['mem'] = (0.11 + 0.44 + 1.01 + 1.23 +
#                                               0.5) * 8  #, 0]

# ENERGY_TABLE['PIM'][PIMType.BA]['sram'] = 0.0034
# ENERGY_TABLE['PIM'][PIMType.BG]['sram'] = 0.0034
# ENERGY_TABLE['PIM'][PIMType.BUFFER]['sram'] = 0.0034

# ENERGY_TABLE['PIM'][PIMType.BA]['alu'] = 0.32
# ENERGY_TABLE['PIM'][PIMType.BG]['alu'] = 0.32
# ENERGY_TABLE['PIM'][PIMType.BUFFER]['alu'] = 0.32

# ENERGY_TABLE['PIM'][PIMType.BA]['io'] = [0.3, 0.5, 1.23, 1.01] 
# ENERGY_TABLE['PIM'][PIMType.BG]['io'] = [0.3, 0.5, 1.23, 1.01]
# ENERGY_TABLE['PIM'][PIMType.BUFFER]['io'] = [0.3, 0.5, 1.23, 1.01] 

# # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
# ENERGY_TABLE['PIM'][PIMType.BA]['comm'] = 10.4
# ENERGY_TABLE['PIM'][PIMType.BG]['comm'] = 10.4
# ENERGY_TABLE['PIM'][PIMType.BUFFER]['comm'] = 10.4


def make_gpu_config(
                    num_gpu=None,
                    hbf_en=False,
):
    config = {}
    config["HBF_EN"] = hbf_en
    config["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu
    config["FLOPS_PER_DEVICE"] = 4.5*1000*1000 *1000 *1000* 1000 #4500TFLOPS
    config['MEM_CAPACITY_PER_DEVICE'] = 2048 *1024 * 1024 * 1024 if hbf_en else 180 *1024 *1024*1024
    config['GPU_COMM_BANDWIDTH'] = 900 * 1024 *1024 * 1024
    config['CPU_COMM_BANDWIDTH'] = 25 * 1024 *1024 * 1024
    config['GPU_HBF_BANDWIDTH'] =  4 * 1.6 * 1024 * 1024 *1024 * 1024
    config['GPU_HBM_BANDWIDTH'] = 4 * 1024 * 1024 * 1024 * 1024 if hbf_en else 8 * 1024 * 1024 * 1024 * 1024
    config['ENERGY_TABLE'] = ENERGY_TABLE['GPU']

    #if gpu_type == GPUType.A100a:
        # Ref: DGX-A100 whitepaper
        # config['GPU']["NUM_CORE"] = 108
        # config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
        #                                     if flops is None else flops
        # config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
        #                                             if mem_cap is None else mem_cap

        # config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
        #                                           if mem_bw is None else mem_bw
        # config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 3.8 * 1000 * 1000 * 1000 * 1000
        # config['GPU']["L1_CAP_PER_CORE"] = 192 * 1024
        # config['GPU']["L2_CAP_PER_DEVICE"] = 40 * 1024 * 1024
        # config['GPU']["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
        # config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # config['CPU']["NUM_DEVICE"] = 2
        # config['CPU']["NUM_CORE"] = 64
        # config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        # config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        # config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 200 * 1000 * 1000 * 1000
        # config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # # TODO: Modify it
        # config['CPU']["L1_CAP_PER_CORE"] = 96 * 1024
        # config['CPU']["L2_CAP_PER_DEVICE"] = 256 * 1024 * 1024
        # config['CPU']["INTERFACE_BW"] = 4 * 64 * 1000 * 1000 * 1000
        # config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']
        

    return config


# Rank x BG x BA / 2 (tCCD)
# BW_SCALE = {
#     False: {
#         PIMType.BA: 2 * 4 * 4 / 2,
#         PIMType.BG: 2 * 4,
#         PIMType.BUFFER: 1
#     },
#     True: {
#         PIMType.BA: 9,
#         PIMType.BG: 3,
#         PIMType.BUFFER: 1
#     }
# }


def make_pim_config(
                    num_pim_device=2,
                    num_pim_stack=4,
                    sparse_enable=False,):
    config= {}
    n_ck = 1/4 * 1e-9 #s
    config["SPARSE_ENABLE"] = sparse_enable
    config["NUM_PIM_DEVICE"] = num_pim_device
    config["NUM_PIM_STACK"] = 4
    config["NUM_PCH_PER_STACK"] = 32
    config["NUM_BG_PER_PCH"] = 16
    config["NUM_ROW"] = 8192
    config["n_compute_per_row"] = 32
    config["t_row"] = 70 * n_ck
    config["t_compute"] = 8 * n_ck
    config["ENERGY_TABLE"] = ENERGY_TABLE['SparsePIM']
    return config

# def make_sparse_pim_config(
#                     num_pim_device=2,
#                     num_pim_stack=4,
#                     bw_scale=None,
#                     power_constraint=False):
#     config= {}
#     n_ck = 1/4 * 1e-9 #s

#     config["ENERGY_TABLE"] = ENERGY_TABLE['SparsePIM']
#     config["NUM_PIM_DEVICE"] = num_pim_device
#     config["NUM_PIM_STACK"] = 4
#     config["NUM_PCH_PER_DEVICE"] = 32
#     config["NUM_BG_PER_PCH"] = 16
#     config["NUM_ROW"] = 8192
#     config["n_compute_per_row"] = 32
#     config["t_row"] = 34 * n_ck
#     config["t_compute"] = 2 * n_ck

#     return config

# def make_pim_config(pim_type: PIMType,
#                     interface_type: InterfaceType,
#                     opb=1,
#                     num_attacc=8,
#                     num_hbm=5,
#                     bw_scale=None,
#                     power_constraint=False):
#     config = {}


#     config["ENERGY_TABLE"] = ENERGY_TABLE['PIM'][pim_type]

#     internal_bandwidth_scale =  BW_SCALE[power_constraint][pim_type] \
#                                 if bw_scale is None else bw_scale
#     config["NUM_ATTACC"] = num_attacc
#     config["NUM_HBM"] = num_hbm
#     config["MEM_CAPACITY_PER_HBM"] = 16 * 1024 * 1024 * 1024
#     config[
#         "MEM_BW_PER_HBM"] = 670.4 * 1000 * 1000 * 1000 * internal_bandwidth_scale
#     config["FLOPS_PER_HBM"] = config["MEM_BW_PER_HBM"] * opb
#     config["SOFTMAX_MEM_BW"] = 670.4 * 1000 * 1000 * 1000 * num_hbm
#     config["SOFTMAX_FLOPS"] = config["SOFTMAX_MEM_BW"]

#     if interface_type == InterfaceType.NVLINK3:
#         config["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
#     elif interface_type == InterfaceType.NVLINK4:
#         config["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
#     elif interface_type == InterfaceType.PCIE4:
#         config["INTERFACE_BW"] = 64 * 1000 * 1000 * 1000
#     elif interface_type == InterfaceType.PCIE5:
#         config["INTERFACE_BW"] = 128 * 1000 * 1000 * 1000
#     else:
#         assert 0, "Invalid interface type"

#     return config


# def make_model_config(name, dtype):
#     model_table = {}
#     model_table['GPT-175B'] = [96, 12288, 96, 128, 4, 1]
#     model_table['GPT-89B'] = [48, 12288, 96, 128, 4, 1]
#     model_table['GPT-13B'] = [40, 5120, 40, 128, 4, 1]
#     model_table['LLAMA-7B'] = [32, 4096, 32, 128, 8 / 3, 1]
#     model_table['LLAMA-65B'] = [80, 8192, 64, 128, 8 / 3, 1]
#     model_table['MT-76B'] = [60, 10240, 40, 128, 4, 1]
#     model_table['MT-146B'] = [80, 12288, 80, 128, 4, 1]
#     model_table['MT-310B'] = [96, 16384, 128, 128, 4, 1]
#     model_table['MT-530B'] = [105, 20480, 128, 160, 4, 1]
#     model_table['MT-1008B'] = [128, 25600, 160, 160, 4, 1]
#     model_table['OPT-66B'] = [64, 9216, 72, 128, 4, 1]

#     ndec, hdim, nheads, dhead, ff_scale, gqa_size = model_table[name]
#     config = {
#         'name': name,
#         'ndec': ndec,
#         'hdim': hdim,
#         'num_heads': nheads,
#         'dhead': dhead,
#         'ff_scale': ff_scale,
#         'gqa_size': gqa_size,
#         'dtype': dtype
#     }
#     return config
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

def make_model_config(name, model_name):
    config = {}
    if model_name == 'llama4':
        config['name'] = 'llama4'
        config['n_block'] = 48
        config['n_experts'] = 128
        config['experts_per_token'] = 2
        config['n_q_head'] = 40
        config['dim'] = 5120
        config['moe_dim'] = 5120
        config['hdim'] = 16384
        config['hdim_moe'] = 8192
        config['cluster_size'] = 16
        config['n_kv_head'] = 8
        config['dhead'] = 128
    else:
        config['name'] = 'qwen3'
        config['n_block'] = 62
        config['n_experts'] = 160
        config['experts_per_token'] = 8
        config['n_q_head'] = 96
        config['dim'] = 6144
        config['hdim'] = 6144
        config['dim_moe'] = 2560
        config['hdim_moe'] = 6144
        config['cluster_size'] = 16
        config['n_kv_head'] = 8
        config['dhead'] = 128        
        
    return config
    