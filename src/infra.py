## Define models and layer.
## Generate models
from ast import Pass
from src.type import *
from src.model import *
from src.spat import *
from src.stats import *
from src.request import *
from src.evaluate import GPU, PIM
import copy
import random
import numpy as np
import math

class System:

    def __init__(self, energy_stats, latency_stats, tbt_stats, modelinfos=None, hardware_config=None,request_stream:Request_Stream=None, pim_profile_table:PIM_Profile_Table=None, \
        hbf_track_table:HBF_Track_Table=None, warmup_iteration=1024, scheduling_interval=64, scheduling_enable=False, dynamic_enable=False):
        self.energy_stats = energy_stats
        self.latency_stats = latency_stats
        self.tbt_stats = tbt_stats
        self.hardware_config = hardware_config
        self.modelinfos = modelinfos
        self.request_batch = request_stream.request_batch
        self.request_stream = request_stream
        self.n_block = modelinfos["n_block"]
        self.device = hardware_config["device"]
        self.n_device = hardware_config["n_device"]
        self.parallel_degree = hardware_config["n_device"]
        self.balance_history = []
        self.var_history = []
        self.pim_profile_table = pim_profile_table
        self.hbf_track_table = hbf_track_table
        self.warmup_iteration = warmup_iteration
        self.scheduling_interval = scheduling_interval
        self.offloading_ratio = 0
        self.dynamic_enable = dynamic_enable
        self.scheduling_enable = scheduling_enable
        self.GPU = None


    def get_offloading_ratio(self):
        if self.dynamic_enable is False:
            model_mem = 2 *480 * 1e9
        else:
            model_mem = 2 *402 * 1e9
        if self.dynamic_enable is False:
            kv_cache_mem = 4 * self.modelinfos["n_block"] * self.modelinfos["n_kv_head"] * self.modelinfos["dhead"] * self.modelinfos["cluster_size"] * sum(req["total_clusters"] for req in self.request_batch.request.values())
        else:
            kv_cache_mem = 4 * self.modelinfos["n_block"] * self.modelinfos["n_kv_head"] * self.modelinfos["dhead"] * self.modelinfos["cluster_size"] * sum(req["n_cluster"] for req in self.request_batch.request.values())

        gpu_mem = self.hardware_config["GPU"]["MEM_CAPACITY_PER_DEVICE"] * self.hardware_config["n_device"]
        if (kv_cache_mem + model_mem) < gpu_mem:
            self.offloading_ratio = 0
        else:
            self.offloading_ratio = (kv_cache_mem-(gpu_mem - model_mem)) / kv_cache_mem
            
        if self.GPU is not None:
            self.GPU.offloading_ratio = self.offloading_ratio
        #self.offloading_ratio = 0
        print(f"offloading_ratio: {self.offloading_ratio}, kv_cache_mem: {kv_cache_mem/1024/1024/1024}GB, model_mem: {model_mem/1024/1024/1024}GB, gpu_mem: {gpu_mem/1024/1024/1024}GB")

        
    def transformer_block_build(self, moe_enable=True):
        return Transformer(self.device, self.modelinfos, self.request_batch, moe_enable, self.parallel_degree).build()

    def hardware_setup(self):
        print(f'hardware_setup_start')
        self.get_offloading_ratio()
        if self.device == DeviceType.GPU:
            self.GPU = GPU(DeviceType.GPU, self.hardware_config["GPU"], self.request_batch, self.offloading_ratio)
            self.PIM = None
        elif self.device == DeviceType.HB_PIM:
            self.GPU = GPU(DeviceType.GPU, self.hardware_config["GPU"], self.request_batch, self.offloading_ratio)
            self.PIM = PIM(DeviceType.HB_PIM, self.hardware_config["PIM"], self.request_batch)
        print(f"hardware_setup_end")
    def system_setup(self):
        self.hardware_setup()
        self.moe_model = self.transformer_block_build(moe_enable=True)
        self.ffn_model = self.transformer_block_build(moe_enable=False)
    
    def sim(self, max_iteration = None):
        now_time = 0
        last_update_time = 1
        n_iteration = 0
        warmup_finish = 0 if self.warmup_iteration > 0 else 1
        print(f"start simulation")
        while True:
            if warmup_finish == 0:
                "start warmup"
                for i in range(self.warmup_iteration):
                    #print(f"warmup iteration {i}")
                    cluster_mapping_table = self.pim_profile_table.cluster_mapping_table
                    hit_clusters_counts = 0
                    miss_clusters_counts = 0
                    activated_clusters_table = self.request_batch.gen_activated_clusters()
                    
                    for request_id, clusters in activated_clusters_table.items():
                        for cluster_id in clusters:
                            # default kv_head_id is 0
                            key = (request_id, 0, int(cluster_id))
                            if key in cluster_mapping_table: # PIM cluster hit
                                stack, pch, bg, _ = cluster_mapping_table[key]
                                self.pim_profile_table.update(request_id, 0, cluster_id)
                                hit_clusters_counts += 1
                            else: # HBF cluster miss
                                miss_clusters_counts += 1
                                self.hbf_track_table.update(request_id, 0, cluster_id)
                    print(f"warmup iteration {i}, hit_clusters_counts: {hit_clusters_counts}, miss_clusters_counts: {miss_clusters_counts}, hit_ratio: {hit_clusters_counts / (hit_clusters_counts + miss_clusters_counts)}")

                    if i % self.scheduling_interval == 0:
                        self.pim_profile_table.leakage_average()
                        self.hbf_track_table.leakage_average()
                        average_utilization, _, total_variance = self.pim_profile_table._get_balance_meta()
                        self.var_history.append((average_utilization, total_variance, 0)) # 0: no scheduling, 1: scheduling
                        _,n_promotions = promotion(self.hbf_track_table, self.pim_profile_table, 4096)
                        print(f"n_promotions: {n_promotions}")
                        print(f"request_batch: {len(self.request_batch.request)}, average_utilization: {average_utilization}, total_variance: {total_variance}")

                        n_swaps = 0
                        for error_threshold in [0.4, 0.2, 0.1, 0.05]:
                            initial_variance, final_variance, balance_history, swaps = self.pim_profile_table.greedy_balance_load(error_threshold)
                            n_swaps += swaps
                            self.balance_history.append([balance_history, error_threshold])
                        self.var_history.append((average_utilization, total_variance, 1))
                        print(f"n_swaps: {n_swaps}, variance: {initial_variance}, final_variance: {final_variance}")

                warmup_finish = 1
                

            # if warmup_finish == 0:
            #     print(f"warmup iteration {n_iteration}")
            #     if n_iteration >= self.warmup_iteration:
            #         warmup_finish = 1
            #         n_iteration = 0
            #         print(f"warmup finish")
            # else:
            #     print(f"iteration {n_iteration}")
            else:
                moe_en = 1
                layer_wise_sim = 0
                # layer-wise simulation
                if layer_wise_sim == 1:
                    # for block_id in range(self.n_block):

                    #     if self.modelinfos["name"] == "llama4":
                    #         moe_en = ~moe_en

                    #     if moe_en == 0:
                    #         model = self.ffn_model
                    #     else:
                    #         model = self.moe_model

                    #     for layer in model:
                    #         if layer.type == "spat":
                    #             if self.device == DeviceType.GPU:
                    #                 energy, latency = self.GPU.execute(layer.get_flops(), layer.get_size())
                    #                 if
                    #                 Energy["gpu_spat"] += energy
                    #                 Latency["gpu_spat"] += latency
                    #             elif self.device == DeviceType.PIM:
                    #                 energy, latency = self.PIM.execute(layer.get_flops(), layer.get_size(),self.request_stream)
                    #                 Energy["pim_spat"] += energy
                    #                 Latency["pim_spat"] += latency
                    #             elif self.device == DeviceType.Sparse_PIM:
                    #                 energy, latency = self.Sparse_PIM.execute(layer.get_flops(), layer.get_size(),self.request_stream)
                    #                 Energy["pim_spat"] += energy
                    #                 Latency["pim_spat"] += latency

                    #         elif layer.type == "comm":
                    #             energy, latency = self.GPU.execute(layer.get_flops(), layer.get_size())
                    #             Energy["comm"] += energy
                    #             Latency["comm"] += latency
                    #         else: # FC layer
                    #             energy, latency = self.GPU.execute(layer.get_flops(), layer.get_size())
                    #             Energy["fc"] += energy
                    #             Latency["fc"] += latency
                    #             Energy[layer.name] += energy
                    #             Latency[layer.name] += latency
                    Pass
                else:
                ## fast simulation
                    if self.modelinfos["name"] == "llama4":
                        iteration_energy = 0
                        iteration_latency = 0
                        iteration_mem_energy = 0
                        iteration_compute_energy = 0    
                        for layer in self.moe_model:
                            if layer.type == LayerType.SpAt_Score_Context or layer.type == LayerType.SpAt_Similarity:
                                if self.device == DeviceType.GPU and warmup_finish == 1:
                                    energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                else: # self.device == DeviceType.PIM
                                    energy, latency, mem_energy, compute_energy = self.PIM.execute(layer,self.request_batch, self.pim_profile_table, self.hbf_track_table, self.GPU)
                                if warmup_finish == 1:
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2
                            else: # self.device == DeviceType.PIM
                                if warmup_finish == 1:
                                    energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2
                        for layer in self.ffn_model:
                            if layer.type == LayerType.SpAt_Score_Context or layer.type == LayerType.SpAt_Similarity:
                                if self.device == DeviceType.GPU:
                                    energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                else: # self.device == DeviceType.PIM
                                    energy, latency, mem_energy, compute_energy = self.PIM.execute(layer,self.request_batch, self.pim_profile_table, self.hbf_track_table, self.GPU)
                                if warmup_finish == 1:
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2
                            else: # self.device == DeviceType.PIM
                                if warmup_finish == 1:
                                    energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2
                    else: # qwen-3
                        iteration_energy = 0
                        iteration_latency = 0
                        iteration_mem_energy = 0
                        iteration_compute_energy = 0    
                        for layer in self.moe_model:
                            if layer.type == LayerType.SpAt_Score_Context or layer.type == LayerType.SpAt_Similarity:
                                if self.device == DeviceType.GPU:
                                    energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                else: # self.device == DeviceType.PIM
                                    energy, latency, mem_energy, compute_energy = self.PIM.execute(layer,self.request_batch, self.pim_profile_table, self.hbf_track_table, self.GPU)
                                if warmup_finish == 1:
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2
                            else: # self.device == DeviceType.GPU
                                energy, latency, mem_energy, compute_energy = self.GPU.execute(layer,self.request_batch)
                                if warmup_finish == 1:
                                    self.energy_stats[layer.name] += energy * self.n_block / 2
                                    self.latency_stats[layer.name] += latency * self.n_block / 2
                                    self.energy_stats["mem_energy"] += mem_energy * self.n_block / 2
                                    self.energy_stats["compute_energy"] += compute_energy * self.n_block / 2
                                    self.energy_stats["sum"] += energy * self.n_block / 2
                                    self.latency_stats["sum"] += latency * self.n_block / 2
                                    iteration_mem_energy += mem_energy * self.n_block / 2 
                                    iteration_compute_energy += compute_energy * self.n_block / 2
                                    iteration_energy += energy * self.n_block / 2
                                    iteration_latency += latency * self.n_block / 2

                n_iteration += 1
                if self.device != DeviceType.GPU:
                    average_utilization, _, total_variance = self.pim_profile_table._get_balance_meta()
                    self.tbt_stats.append((len(self.request_batch.request), iteration_latency, average_utilization, total_variance/average_utilization, n_iteration % self.scheduling_interval))
                else:
                    self.tbt_stats.append((len(self.request_batch.request), iteration_latency, 0, 0, 0))

                print(f"iteration {n_iteration}, request_batch_size: {len(self.request_batch.request)}, iteration_energy: {iteration_energy}, iteration_mem_energy: {iteration_mem_energy}, iteration_latency: {iteration_latency}")
                request_to_exit = self.request_batch.update()
                if len(request_to_exit) and self.device != DeviceType.GPU:
                    print(f"request_to_exit: {request_to_exit}")
                    for request_id in request_to_exit:
                        self.pim_profile_table.request_exit(request_id)
                        #self.hbf_track_table.request_exit(request_id)

                if self.device != DeviceType.GPU and n_iteration % self.scheduling_interval == 0:
                    self.pim_profile_table.leakage_average()
                    average_utilization, _, total_variance = self.pim_profile_table._get_balance_meta()
                    self.var_history.append((average_utilization, total_variance, 0)) # 0: no scheduling, 1: scheduling
                    print(f"request_batch: {len(self.request_batch.request)}, average_utilization: {average_utilization}, total_variance: {total_variance}")
                    for request_id in self.hbf_track_table.keys():
                        self.hbf_track_table[request_id].leakage_average()
                        _,n_promotions = promotion(self.hbf_track_table[request_id], self.pim_profile_table, 4096)
                        print(f"n_promotions: {n_promotions}")
                    if warmup_finish == 1:
                        self.energy_stats["promotion"] += 2 * n_promotions * (2*2*128*16) * (self.GPU.energy_table['hbm'] + self.GPU.energy_table['hbm']) * self.n_device
                        self.latency_stats["promotion"] +=  n_promotions * (2*2*128*16) / self.GPU.hbf_memory_bandwidth + n_promotions * (2*2*128*16) / self.GPU.hbm_memory_bandwidth


                    if self.scheduling_enable:
                        n_swaps = 0
                        for error_threshold in [0.4, 0.2, 0.1, 0.05]:
                            initial_variance, final_variance, balance_history, swaps = self.pim_profile_table.greedy_balance_load(error_threshold)
                            n_swaps += swaps
                            self.balance_history.append([balance_history, error_threshold])
                        self.var_history.append((average_utilization, total_variance, 1))
                        if warmup_finish == 1:
                            # TODO: more accurate energy and latency calculation
                            self.energy_stats["balance"] += 2 * n_swaps * (2*4*1024*8) * (self.GPU.energy_table['hbm'] + self.GPU.energy_table['hbm']) * self.n_device
                            self.latency_stats["balance"] += 2 * 2 * n_swaps * (2*4*1024*8) / self.GPU.hbm_memory_bandwidth
                        print(f"n_swaps: {n_swaps}, variance: {initial_variance}, final_variance: {final_variance}")


            # dynamic arrival request
            if self.dynamic_enable:
                now_time += iteration_latency*self.modelinfos["cluster_size"]
                if now_time > last_update_time + 1:
                    n_new_requests = self.request_stream.add_new_requests()
                    print(f"Add n_new_requests: {n_new_requests}")
                    last_update_time = now_time


            self.get_offloading_ratio()


            if len(self.request_batch.request) == 0:
                return self.energy_stats.copy(), self.latency_stats.copy(), self.tbt_stats.copy()


            if max_iteration is not None and n_iteration >= max_iteration:
                return self.energy_stats.copy(), self.latency_stats.copy(), self.tbt_stats.copy()
