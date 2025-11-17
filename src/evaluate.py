## Define models and layer.
## Generate models
from src.type import *
from src.config import *
from src.spat import *
import copy
import random
import numpy as np
import math
from collections import defaultdict

class GPU:

    def __init__(self, name: DeviceType, config, request_batch=None, offloading_ratio=0):
        self.name = name
        self.num_gpu = config['NUM_DEVICE']
        self.hbf_en = config['HBF_EN']
        self.peak_flops = config['FLOPS_PER_DEVICE']
        self.peak_hbm_bandwidth = config['GPU_HBM_BANDWIDTH']
        self.peak_hbf_bandwidth = config['GPU_HBF_BANDWIDTH']
        self.aggregate_memory_capacity = config['MEM_CAPACITY_PER_DEVICE'] * self.num_gpu
        self.gpu_comm_bandwidth = config['GPU_COMM_BANDWIDTH']
        self.cpu_comm_bandwidth = config['CPU_COMM_BANDWIDTH']
        self.energy_table = config['ENERGY_TABLE']
        self.hbf_memory_bandwidth = config['GPU_HBF_BANDWIDTH'] if self.hbf_en else 0
        self.hbm_memory_bandwidth = config['GPU_HBM_BANDWIDTH']
        self.peak_memory_bandwidth = self.hbf_memory_bandwidth if self.hbf_en else self.hbm_memory_bandwidth
        self.offloading_ratio = offloading_ratio
        self.gpu_mem_energy = config['ENERGY_TABLE']['hbf'] if self.hbf_en else config['ENERGY_TABLE']['hbm']
        #self.mem_init_latency = 3*1e-6 if self.hbf_en else 26.4* 1e-9 # 3us HBF tR  /// (29 + 16/4) DRAM latency tras + tfaw/4
        self.mem_init_latency = 0
    def execute(self, layer, request_batch=None, block_id=None):
        #print(f"GPU execute layer: {layer.name}, type: {layer.type}")
        if request_batch is not None and layer.type == LayerType.SpAt_Score_Context:
            activated_clusters_table = request_batch.gen_activated_clusters()
            # for kv_head_id in range(layer.n_kv_head):
            #     activated_clusters_table[kv_head_id] = request_batch.gen_activated_clusters(block_id, kv_head_id)
            #     #print(f"activated_clusters_table[{kv_head_id}]: {activated_clusters_table[kv_head_id]}")

        m, n, k= layer.get_infos()
        operation_intensity = m
        flops = layer.get_flops()
        in1, in2, out = layer.get_size()
        execute_time = 0
        energy = 0
        mem_energy = 0
        compute_energy = 0
        if layer.type in [LayerType.FC, LayerType.MATMUL, LayerType.NORM, LayerType.ACT]:
            execute_time = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops) + self.mem_init_latency
            gpu_mem_energy = ((in1+out)*self.energy_table['hbm'] + in2*self.energy_table['hbf']) if self.hbf_en else ((in1+in2+out)*self.energy_table['hbm'])
            gpu_alu_energy = flops/2*self.energy_table['alu']
            gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2'])*4 # 4: 4 is scaling factor
            energy = (gpu_mem_energy + gpu_alu_energy + gpu_onchip_mem_energy) *  self.num_gpu
            mem_energy = gpu_mem_energy *  self.num_gpu
            compute_energy = (gpu_alu_energy + gpu_onchip_mem_energy) *  self.num_gpu
        elif layer.type in [LayerType.ALL_GATHER, LayerType.ALL_REDUCE]:
            execute_time = (in1+in2+out)/self.gpu_comm_bandwidth
            gpu_comm_energy = (in1+in2+out)*(self.energy_table['comm'] + self.energy_table['hbm'])
            gpu_mem_energy = (in1+in2+out)*self.energy_table['hbm']
            gpu_alu_energy = flops/2*self.energy_table['alu']
            gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2'])*4 # 4: 4 is scaling factor
            energy = (gpu_mem_energy + gpu_alu_energy + gpu_onchip_mem_energy + gpu_comm_energy) *  self.num_gpu
            mem_energy = gpu_mem_energy *  self.num_gpu
            compute_energy = (gpu_alu_energy + gpu_onchip_mem_energy + gpu_comm_energy) *  self.num_gpu
        elif layer.type == LayerType.SpAt_Similarity:
            # head_latency = [0 for _ in range(layer.n_kv_head)]
            # head_energy = [0 for _ in range(layer.n_kv_head)]            
            # for kv_head_id in range(layer.n_kv_head):
            #     for request_id in layer.request_batch.request:
            #         m, n, k= layer.get_infos(request_id)
            #         operation_intensity = m
            #         flops = layer.get_flops()
            #         in1, in2, out = layer.get_size()
            #         req_execute_time = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops)
            #         req_gpu_mem_energy = ((in1+out)*self.energy_table['hbm'] + in2*self.energy_table['hbf']) if self.hbf_en else (in1+in2+out)*self.energy_table['hbm']
            #         req_gpu_alu_energy = flops/2*self.energy_table['alu']
            #         req_gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2']) *4 # 4: 4 is scaling factor
            #         req_energy = (req_gpu_mem_energy + req_gpu_alu_energy + req_gpu_onchip_mem_energy) 
            #         head_latency[kv_head_id] += req_execute_time
            #         head_energy[kv_head_id] += req_energy
            #         mem_energy += req_gpu_mem_energy 
            #         compute_energy += (req_gpu_alu_energy + req_gpu_onchip_mem_energy) 


            ### 1 head for 1 device       
            for request_id in layer.request_batch.request:
                m, n, k= layer.get_infos(request_id)
                operation_intensity = m
                flops = layer.get_flops()
                in1, in2, out = layer.get_size()
                req_execute_time = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops)
                req_gpu_mem_energy = ((in1+out)*self.energy_table['hbm'] + in2*self.energy_table['hbf']) if self.hbf_en else (in1+in2+out)*self.energy_table['hbm']
                req_gpu_alu_energy = flops/2*self.energy_table['alu']
                req_gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2']) *4 # 4: 4 is scaling factor
                req_energy = (req_gpu_mem_energy + req_gpu_alu_energy + req_gpu_onchip_mem_energy) 
                execute_time += req_execute_time
                energy += req_energy * self.num_gpu
                mem_energy += req_gpu_mem_energy * self.num_gpu
                compute_energy += (req_gpu_alu_energy + req_gpu_onchip_mem_energy) * self.num_gpu

        elif layer.type ==LayerType.SpAt_Score_Context:
            for request_id, request_info in layer.request_batch.request.items():
                #print(f"request_id: {request_id}")
                m, n, k= layer.get_infos(request_id)
                operation_intensity = m
                flops = layer.get_flops()
                in1, in2, out = layer.get_size()
                total_activated_clusters = in2 / layer.cluster_size
                hbm_miss_clusters_counts = 0
                req_kv_head_activated_clusters = activated_clusters_table
                #print(f"request_info: {req_kv_head_activated_clusters}")
                #print(f"layer.request_id: {layer.request_id}, kv_head_id: {kv_head_id}, req_kv_head_activated_clusters: {req_kv_head_activated_clusters}")

                # 判断是否需要CPU-Offloading
                if self.offloading_ratio > 0:
                    clusters_in_cpu = 0
                    for request_id, clusters in req_kv_head_activated_clusters.items():
                        clusters_in_cpu += (clusters> (request_info["total_cluster"] * self.offloading_ratio) ).sum()
                    #print(f"clusters_in_cpu: {clusters_in_cpu}")
                    cpu_latency = clusters_in_cpu * layer.cluster_size * layer.head_dim / self.cpu_comm_bandwidth
                    cpu_energy = clusters_in_cpu * layer.cluster_size * layer.head_dim * (self.energy_table['comm'] + self.energy_table['dimm'] + self.energy_table['hbm'])
                    hbm_miss_clusters_counts +=1
                else:
                    cpu_latency = 0
                    cpu_energy = 0
                gpu_hit_clusters_counts = total_activated_clusters - hbm_miss_clusters_counts
                gpu_mem_energy = ((in1+out)*self.energy_table['hbm'] + in2*self.energy_table['hbf']) if self.hbf_en else ((in1+in2+out)*self.energy_table['hbm'])
                gpu_alu_energy = flops/2*self.energy_table['alu']
                gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2']) *4 # 4: 4 is scaling factor
                compute_energy = gpu_mem_energy + gpu_alu_energy + gpu_onchip_mem_energy
                #compute_latency = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops)
                compute_latency = (math.ceil(gpu_hit_clusters_counts/8/32)*4*0.8+2.5*0.8)*1e-9 + hbm_miss_clusters_counts * layer.cluster_size / self.peak_memory_bandwidth
                execute_time += (cpu_latency + compute_latency)
                energy += (compute_energy + cpu_energy) * self.num_gpu
                mem_energy += gpu_mem_energy * self.num_gpu
                compute_energy += (gpu_alu_energy + gpu_onchip_mem_energy) * self.num_gpu
            #print(f"layer.name: {layer.name}, layer.type: {layer.type}, energy: {energy}, execute_time: {execute_time}")
        elif layer.type == LayerType.SpAt_Softmax:

            for request_id in layer.request_batch.request.keys():
                m, n, k= layer.get_infos(request_id)
                operation_intensity = m
                flops = layer.get_flops()
                in1, in2, out = layer.get_size()
                req_execute_time = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops)
                req_gpu_mem_energy = (in1+in2+out)*self.energy_table['hbm']
                req_gpu_alu_energy = flops/2*self.energy_table['alu']
                req_gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2'])*4 # 4: 4 is scaling factor
                req_energy = (req_gpu_mem_energy + req_gpu_alu_energy + req_gpu_onchip_mem_energy)
                execute_time += req_execute_time
                energy += req_energy * self.num_gpu
                mem_energy += req_gpu_mem_energy * self.num_gpu
                compute_energy += (req_gpu_alu_energy + req_gpu_onchip_mem_energy) * self.num_gpu
                    

        
        return energy, execute_time, mem_energy, compute_energy


class PIM:

    def __init__(self, name: DeviceType, config, request_batch=None):
        self.name = name
        self.sparse_enable = config['SPARSE_ENABLE']
        self.num_pim_device = config['NUM_PIM_DEVICE']
        self.num_pim_stack = config['NUM_PIM_STACK']
        self.num_pch_per_stack = config['NUM_PCH_PER_STACK']
        self.num_bg_per_pch = config['NUM_BG_PER_PCH']
        self.num_row = config['NUM_ROW']
        self.energy_table = config['ENERGY_TABLE']
        self.e_row = ENERGY_TABLE['SparsePIM']['row']
        self.e_read = ENERGY_TABLE['SparsePIM']['read']
        self.e_compute = ENERGY_TABLE['SparsePIM']['compute']
        self.n_compute_per_row = 32 if self.sparse_enable else 2
        self.t_compute = config['t_compute']
        self.t_row = config['t_row']
        self.n_bk_per_bg = 4
        self.n_bk_per_pch = 64

    def execute(self, layer, request_batch=None, pim_profile_table:PIM_Profile_Table=None, hbf_track_table:HBF_Track_Table=None, GPU=None, block_id=0, test_mode=False):
        #print(f"PIM execute layer: {layer.name}, type: {layer.type}")
        mem_energy, compute_energy = 0, 0

        if layer.type == LayerType.SpAt_Score_Context:

            pim_activated_table=[[[0 for _ in range(self.num_bg_per_pch)] for _ in range(self.num_pch_per_stack)] for _ in range(self.num_pim_stack)]
            cluster_mapping_table = pim_profile_table.cluster_mapping_table
            missing_clusters = defaultdict(lambda: defaultdict(list))
            missing_clusters_counts = 0
            hit_clusters_counts = 0
            activated_clusters_table = request_batch.gen_activated_clusters()
            n_requests = len(activated_clusters_table)
            #print(f"n_requests: {n_requests}")
            # if request_batch is not None:
            #     for kv_head_id in range(layer.n_kv_head):
            #         activated_clusters_table[kv_head_id] = request_batch.gen_activated_clusters(block_id, kv_head_id)

            # # PIM cluster hit
            # # for kv_head_id, request_clusters in activated_clusters_table.items():
            # #     if kv_head_id < (8//self.num_pim_device):
            # #         continue #只统计一半的设备
            
            for request_id, clusters in activated_clusters_table.items():
                for cluster_id in clusters:
                    # default kv_head_id is 0
                    key = (request_id, 0, int(cluster_id))
                    if key in cluster_mapping_table: # PIM cluster hit
                        stack, pch, bg, _ = cluster_mapping_table[key]
                        pim_profile_table.update(request_id, 0, cluster_id)
                        #print(f"stack: {stack}, pch: {pch}, bg: {bg}")
                        pim_activated_table[stack][pch][bg] += 1
                        hit_clusters_counts += 1
                    else: # HBF cluster hit
                        hbf_track_table[request_id].update(request_id, 0, cluster_id)
                        missing_clusters[0][request_id].append(cluster_id)
                        missing_clusters_counts += 1


            print(f"missing_clusters_counts: {missing_clusters_counts}, hit_clusters_counts: {hit_clusters_counts}. hit ratio: {hit_clusters_counts / (hit_clusters_counts + missing_clusters_counts)}")
            # GPU execution
            # gpu_flops = layer.get_flops()
            # in1, in2, out = layer.get_size()
            m = layer.m
            n = layer.head_dim
            operation_intensity = m
             # GPU execution
            if missing_clusters_counts > 0:
                gpu_execute_time = 0
                gpu_energy = 0
                for request_id, clusters in missing_clusters[0].items():
                    n = layer.head_dim
                    k = len(clusters) * layer.cluster_size
                    in1=m*n*layer.dbyte
                    in2=n*k*layer.dbyte
                    out=m*k*layer.dbyte
                    gpu_flops = 2*m*n*k

                    gpu_execute_time += gpu_flops/min(operation_intensity * GPU.peak_memory_bandwidth, GPU.peak_flops)
                    gpu_mem_energy = ((in1+out)*GPU.energy_table['hbm'] + in2*GPU.energy_table['hbf'])
                    gpu_alu_energy = gpu_flops/ 2* GPU.energy_table['alu']
                    gpu_onchip_mem_energy = (in1+in2+out)*(GPU.energy_table['reg'] + GPU.energy_table['l1'] + GPU.energy_table['l2']) *4 # 4: 4 is scaling factor
                    gpu_energy += (gpu_mem_energy + gpu_alu_energy + gpu_onchip_mem_energy) * GPU.num_gpu
                    mem_energy += gpu_mem_energy * GPU.num_gpu
                    compute_energy += (gpu_alu_energy + gpu_onchip_mem_energy) * GPU.num_gpu

                
            # PIM execution
            test =0
            if test:
                pim_execute_time = 0
                pim_energy = 0
                pim_bg_latency = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch)]
                pim_bg_energy = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch)]
                pim_pch_latency = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack)]
                pim_pch_energy = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack)]
                for stack in range(self.num_pim_stack):
                    for pch in range(self.num_pch_per_stack):
                        for bg in range(self.num_bg_per_pch):
                            #print(f"pim_activated_table[stack][pch][bg]: {pim_activated_table[stack][pch][bg]}")
                            pim_bg_latency[stack * self.num_pch_per_stack * self.num_bg_per_pch + pch * self.num_bg_per_pch + bg] = pim_activated_table[stack][pch][bg] * (self.t_row + 2*self.t_compute * operation_intensity * 32) # 2: 1PE FOR 2Banks
                            pim_bg_energy[stack * self.num_pch_per_stack * self.num_bg_per_pch + pch * self.num_bg_per_pch + bg] = pim_activated_table[stack][pch][bg] * (self.e_row + self.e_read * 32 + self.e_compute * operation_intensity * 32) * self.n_bk_per_bg
                            pim_pch_latency[stack * self.num_pch_per_stack + pch] += pim_activated_table[stack][pch][bg] * (self.t_row + 2 * self.t_compute * operation_intensity  * 2)
                            pim_pch_energy[stack * self.num_pch_per_stack + pch] += pim_activated_table[stack][pch][bg] * (self.e_row + self.e_read * 2 + self.e_compute * operation_intensity * 2) * self.n_bk_per_pch
                sparse_pim_latency = np.percentile(pim_bg_latency, 50)
                #sparse_pim_latency = max(pim_bg_latency)
                sparse_pim_energy = sum(pim_bg_energy) + GPU.energy_table['hbm']*(2*m*n*n_requests+2*m*hit_clusters_counts*16)
                pim_latency = np.percentile(pim_pch_latency, 50)
                #pim_latency = max(pim_pch_latency) 
                pim_energy = sum(pim_pch_energy) + GPU.energy_table['hbm']*(2*m*n*n_requests+2*m*hit_clusters_counts*16)
                #print(f"pim_latency: {pim_latency}, pim_energy: {pim_energy}, sparse_pim_latency: {sparse_pim_latency}, sparse_pim_energy: {sparse_pim_energy}")
                return pim_latency, pim_energy, sparse_pim_latency, sparse_pim_energy
                
            else:
                if self.sparse_enable: #sparse pim
                    pim_bg_latency = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch)]
                    pim_bg_energy = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch)]
                    for stack in range(self.num_pim_stack):
                        for pch in range(self.num_pch_per_stack):
                            for bg in range(self.num_bg_per_pch):
                                pim_bg_latency[stack * self.num_pch_per_stack * self.num_bg_per_pch + pch * self.num_bg_per_pch + bg] = pim_activated_table[stack][pch][bg] * (self.t_row + 2*self.t_compute * operation_intensity * self.n_compute_per_row) # 2: 1PE FOR 2Banks
                                pim_bg_energy[stack * self.num_pch_per_stack * self.num_bg_per_pch + pch * self.num_bg_per_pch + bg] = pim_activated_table[stack][pch][bg] * (self.e_row + self.e_read * self.n_compute_per_row + self.e_compute * operation_intensity * self.n_compute_per_row) * self.n_bk_per_bg
                    pim_execute_time = np.percentile(pim_bg_latency, 50)
                    #pim_execute_time =  max(pim_bg_latency) 
                    pim_energy = sum(pim_bg_energy) * self.num_pim_device 
                else: # pim
                    pim_pch_latency = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack)]
                    pim_pch_energy = [0 for _ in range(self.num_pim_stack * self.num_pch_per_stack)]
                    for stack in range(self.num_pim_stack):
                        for pch in range(self.num_pch_per_stack):
                            for bg in range(self.num_bg_per_pch):
                                pim_pch_latency[stack * self.num_pch_per_stack + pch] += pim_activated_table[stack][pch][bg] * (self.t_row + 2 * self.t_compute * operation_intensity  * self.n_compute_per_row)
                                pim_pch_energy[stack * self.num_pch_per_stack + pch] += pim_activated_table[stack][pch][bg] * (self.e_row + self.e_read * self.n_compute_per_row + self.e_compute * operation_intensity * self.n_compute_per_row) * self.n_bk_per_pch
                
                    pim_execute_time = max(pim_pch_latency) 
                    pim_energy = sum(pim_pch_energy) * self.num_pim_device 

            execute_time = 2 * max(pim_execute_time, gpu_execute_time + GPU.mem_init_latency)
            energy = 2 * (pim_energy + gpu_energy)
            #print(f"sparse_enable: {self.sparse_enable}, pim_execute_time: {pim_execute_time}, gpu_execute_time: {gpu_execute_time}, pim_energy: {pim_energy}J, gpu_energy: {gpu_energy}J, total_energy: {energy}J")

        elif layer.type == LayerType.SpAt_Similarity:
            n_compute = 0
            for request_id in layer.request_batch.request:
                m, n, k= layer.get_infos(request_id)
                operation_intensity = m
                flops = layer.get_flops()
                in1, in2, out = layer.get_size()
                n_compute += math.ceil(in2/(1024*self.n_bk_per_pch))


            execute_time = n_compute/self.num_pim_stack/self.num_pch_per_stack *  (self.t_row + 2 * self.t_compute * operation_intensity * 32)
            energy = n_compute *  (self.e_row + self.e_read * 32 + self.e_compute * operation_intensity *32) * self.n_bk_per_pch * self.num_pim_device
                # req_execute_time = flops/min(operation_intensity*self.peak_memory_bandwidth, self.peak_flops)
                # req_gpu_mem_energy = ((in1+out)*self.energy_table['hbm'] + in2*self.energy_table['hbf']) if self.hbf_en else (in1+in2+out)*self.energy_table['hbm']
                # req_gpu_alu_energy = flops/2*self.energy_table['alu']
                # req_gpu_onchip_mem_energy = (in1+in2+out)*(self.energy_table['reg'] + self.energy_table['l1'] + self.energy_table['l2']) *4 # 4: 4 is scaling factor
                # req_energy = (req_gpu_mem_energy + req_gpu_alu_energy + req_gpu_onchip_mem_energy) 
                # execute_time += req_execute_time
                # energy += req_energy * self.num_gpu
                # mem_energy += req_gpu_mem_energy * self.num_gpu
                # compute_energy += (req_gpu_alu_energy + req_gpu_onchip_mem_energy) * self.num_gpu
        else:
            energy = 0
            execute_time = 0
    
        # update table
        return energy, execute_time, mem_energy, compute_energy


