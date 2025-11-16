import numpy as np
import math
from scipy.stats import zipf
from src.spat import *
class Request_SpAt_stat():

    def __init__(self, total_cluster, n_block, n_kv_head, seed=321, prob_func="zipf"):
        self.alpha = 1.0
        self.s = 1.1
        self.n_kv_head = n_kv_head
        self.total_cluster = total_cluster
        self.n_block = n_block
        self.activated_prob_table=[[[0 for _ in range(total_cluster)] for _ in range(n_kv_head)] for _ in range(n_block)]
        np.random.seed(seed)
        
        for block in range(n_block):
            for kv_head in range(1):
                seed = seed + 1
                if prob_func == "power_law":
                    # 生成幂律分布的概率
                    indices = np.arange(1, total_cluster + 1)
                    prob = indices ** (-self.alpha)
                    prob = prob / prob.sum()  # 归一化
                    # 随机打乱顺序，实现随机分配
                    prob = np.random.permutation(prob)
                    self.activated_prob_table[block][kv_head] = prob
                    
                elif prob_func == "zipf":
                    # 生成Zipf分布的概率
                    x = np.arange(1, total_cluster + 1)
                    prob = zipf.pmf(x, self.s)
                    prob = prob / prob.sum()  # 归一化
                    # 随机打乱顺序，实现随机分配
                    prob = np.random.permutation(prob)
                    self.activated_prob_table[block][kv_head] = prob
                    
                elif prob_func == "random":
                    # 完全随机分配：生成n_cluster个随机数，然后归一化
                    prob = np.random.rand(total_cluster)
                    prob = prob / prob.sum()  # 归一化
                    self.activated_prob_table[block][kv_head] = prob
                    
                elif prob_func == "uniform":
                    # 均匀分布：每个cluster概率相等
                    prob = np.ones(total_cluster) / total_cluster
                    # 随机打乱顺序（虽然概率相同，但可以随机分配位置）
                    prob = np.random.permutation(prob)
                    self.activated_prob_table[block][kv_head] = prob
                else:
                    raise ValueError(f"Invalid probability function: {prob_func}")
        
    def get_activated_prob(self, block, kv_head, n_clusters):
        # 已存在的cluster归一化
        return self.activated_prob_table[block][kv_head][:n_clusters]/self.activated_prob_table[block][kv_head][:n_clusters].sum()




class Request_Batch:

    def __init__(self, activation_ratio, modelinfos):
        self.n_kv_head = modelinfos["n_kv_head"]
        self.n_block = modelinfos["n_block"]
        self.activation_ratio = activation_ratio
        self.activated_clusters = {}
        self.request = {}

    def append(self, id, input_cluster, total_cluster):
        # id as random seed
        activation_table = Request_SpAt_stat(total_cluster, 1, 1, id)
        self.request[id] = {"n_cluster": input_cluster, "n_activated_clusters": math.ceil(input_cluster*self.activation_ratio), "total_cluster": total_cluster, "activation_table": activation_table}
        return self.request

    def update(self):
        request_to_exit = []
        for request_id in self.request.keys():
            self.request[request_id]["n_cluster"] = self.request[request_id]["n_cluster"] + 1
            self.request[request_id]["n_activated_clusters"] = math.ceil(self.request[request_id]["n_cluster"]*self.activation_ratio)
        # 请求退出
        for request_id in self.request.keys():
            if self.request[request_id]["n_cluster"] >= self.request[request_id]["total_cluster"]:
                request_to_exit.append(request_id)
        for request_id in request_to_exit:
            del self.request[request_id]

        return request_to_exit

    def gen_activated_clusters(self, layer_id=0, kv_head_id=0):
        activated_clusters = {}
        for request_id in self.request.keys():
            ## 生成当前cluster的概率表，进行不放回抽样
            temp_act_table = self.request[request_id]["activation_table"].get_activated_prob(layer_id, kv_head_id, self.request[request_id]["n_cluster"])
            activated_clusters[request_id] = np.random.choice(self.request[request_id]["n_cluster"], self.request[request_id]["n_activated_clusters"], replace=False, p=temp_act_table)
        return activated_clusters





        
class Request_Stream:
    def __init__(self, poisson_lambda, activation_ratio, modelinfos, initial_request_count=None, request_tracking_table=None):
        """
        Args:
            poisson_lambda: 泊松分布参数
            activation_ratio: 激活比例
            modelinfos: 模型信息字典
            initial_request_count: 初始请求数量（可选，如果 poisson_lambda 中没有指定）
        """
        self.poisson_lambda = poisson_lambda
        self.activation_ratio = activation_ratio
        self.modelinfos = modelinfos
        self.request_id_counter = 0
        self.request_tracking_table = request_tracking_table
        self.activation_ratio = activation_ratio
        self.request_batch = Request_Batch(activation_ratio, modelinfos)
        
        # 生成初始请求序列
        self.generate_initial_requests(initial_request_count)
    
    def generate_initial_requests(self, initial_request_count):
        """生成初始请求序列"""
        for _ in range(initial_request_count):
            total_cluster = np.random.randint(256, 4096)  # 4096-16384 之间的随机数
            n_cluster = np.random.randint(total_cluster//2, total_cluster-1)
            #n_cluster = total_cluster // 2  # n_cluster 为 total_cluster 的一半
            self.request_id_counter += 1
            self.request_batch.append(self.request_id_counter, n_cluster, total_cluster)
            if self.request_tracking_table is not None:
                self.request_tracking_table[self.request_id_counter] = HBF_Track_Table(1.5*total_cluster / self.activation_ratio)

    def add_new_requests(self):
        """动态加入新请求，数量遵从泊松分布"""

        # 生成泊松分布的请求数量
        n_new_requests = np.random.poisson(self.poisson_lambda)
        
        # 添加新请求
        for _ in range(n_new_requests):
            total_cluster = np.random.randint(256, 4096)  # 4096-16384 之间的随机数
            n_cluster = np.random.randint(total_cluster//2, total_cluster-1)
            self.request_id_counter += 1
            self.request_batch.append(self.request_id_counter, n_cluster, total_cluster)
            if self.request_tracking_table is not None:
                self.request_tracking_table[self.request_id_counter] = HBF_Track_Table(1.5*total_cluster / self.activation_ratio)
        return n_new_requests
    
    def update(self):
        """
        更新请求队列：
        1. 动态加入新请求（泊松分布）
        2. 更新所有请求的 n_cluster
        3. 移除 n_cluster >= total_cluster 的请求
        """

        request_to_exit = self.request_batch.update()
        for request_id in request_to_exit:
            if self.request_tracking_table is not None:
                del self.request_tracking_table[request_id]
        return self.request_batch, request_to_exit