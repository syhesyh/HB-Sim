## Define models and layer.
## Generate models
from src.type import *
import copy
import random
import numpy as np


class FC_Layer:

    def __init__(self, stage, name, type, parallel_type, parallel_degree, m, n, k):
        self.stage = stage
        self.name = name
        self.type = type
        if parallel_type == 'tensor_row':
            self.m = m 
            self.n = n 
            self.k = k // parallel_degree
        elif parallel_type == 'tensor_col':
            self.m = m 
            self.n = n // parallel_degree
            self.k = k
        else:
            self.m = m 
            self.n = n 
            self.k = k
        self.dbyte = 2 #bf16

    def get_infos(self):
        return self.m, self.n, self.k

    def get_flops(self):
        if self.type == LayerType.NORM:
            return 5 * self.m * self.n

        elif self.type == LayerType.ACT:
            if 'relu' in self.name:
                return 1 * self.m * self.n
            elif 'glu' in self.name:
                return (8 + 1) * self.m * self.n
            else:
                return 8 * self.m * self.n
        elif self.type == LayerType.FC:
            return 2 * self.m * self.n * self.k

        elif self.type == LayerType.MATMUL:
            return 2 * self.m * self.n
        else:
            assert 0, "In Function \"get_flops\": Not support layer type"

    def get_size(self):
        in1 = self.dbyte * self.m * self.n
        in2 = self.dbyte * self.n * self.k
        out = self.dbyte * self.m * self.k

        if self.type in [LayerType.NORM,LayerType.ACT, LayerType.MATMUL]:
            in1 = self.dbyte * self.m * self.n
            in2 = 0
            out = in1

            # For SwiGLU and GeGLU
            if 'glu' in self.name:
                in2 = in1

        # elif self.type == LayerType.NORM:
        #     in1 = self.numOp * self.m * self.n * self.dbyte
        #     in2 = in1
        #     out = in1

        return in1, in2, out


class Attention_Layer:

    def __init__(self, stage, name, type, parallel_type, parallel_degree, n_q_head, n_kv_head, head_dim, cluster_size, request_batch):
        self.stage = stage
        self.request_batch = request_batch
        self.name = name
        self.type = type
        self.head_dim = head_dim
        self.cluster_size = cluster_size
        self.n_q_head = n_q_head
        self.n_kv_head = n_kv_head
        self.q_head_per_kv_head = n_q_head // n_kv_head
        self.head_per_device = n_kv_head // parallel_degree
        self.m = n_q_head // n_kv_head
        self.dbyte = 2
        
    def get_infos(self, request_id=None):
        if request_id is not None:
            n_cluster = self.request_batch.request[request_id]["n_cluster"]
            n_activated_clusters = self.request_batch.request[request_id]["n_activated_clusters"]
        else:
            n_cluster = 0
            n_activated_clusters = 0
        if self.type == LayerType.SpAt_Similarity:
            self.m = self.q_head_per_kv_head
            self.n = self.head_dim
            self.k = n_cluster
        elif self.type == LayerType.SpAt_Score_Context:
            self.m = self.q_head_per_kv_head
            self.n = self.head_dim
            self.k = n_activated_clusters * self.cluster_size
        elif self.type == LayerType.SpAt_Softmax:
            self.m = self.q_head_per_kv_head
            self.n = n_activated_clusters * self.cluster_size
            self.k = 1 # for softmax, we only need to compute the softmax of the score
        return self.m, self.n, self.k

    def get_flops(self):
        if self.type == LayerType.SpAt_Softmax:
            return 5*self.m*self.n
        else:
            return 2*self.m*self.n*self.k

    def get_size(self):
        in1 = self.dbyte * self.m * self.n
        in2 = self.dbyte * self.n * self.k
        out = self.dbyte * self.m * self.k

        if self.type == LayerType.SpAt_Softmax:
            in1 = self.dbyte * self.m * self.n
            in2 = 0
            out = in1

        return in1, in2, out

class MoE_Activation_Stat:

    def __init__(self, stage, n_input_tokens, n_experts=1, experts_per_token=1):
        self.stage = stage
        self.n_input_tokens = n_input_tokens
        self.n_experts = n_experts
        self.experts_per_token = experts_per_token
        self.expert_token_count = {}  # 存储每个专家被激活的token数量 {expert_id: token_count}
        
    def get_activated_experts(self, seed=None):
        """
        计算每个token随机激活experts_per_token个专家后，返回每个激活专家及其被激活的token数量
        
        Args:
            seed: 随机种子，用于可重复性
            
        Returns:
            expert_token_count: 字典，{expert_id: token_count}，表示每个激活的专家被多少个token激活
        """
        if seed is not None:
            random.seed(seed)
        
        self.expert_token_count.clear()
        
        # 为每个token随机选择experts_per_token个专家
        for token_id in range(self.n_input_tokens):
            # 从n_experts个专家中随机选择experts_per_token个（不重复）
            selected_experts = random.sample(range(self.n_experts), self.experts_per_token)
            # 统计每个专家被多少个token激活
            for expert_id in selected_experts:
                self.expert_token_count[expert_id] = self.expert_token_count.get(expert_id, 0) + 1
        
        # 返回每个激活专家及其被激活的token数量
        return self.expert_token_count.copy()

class Communication:

    def __init__(self, stage, name, type, parallel_degree, m, n, k):
        self.stage = stage
        self.name = name
        self.m = m
        self.n = n
        self.k = k
        self.type = type
        # self.parallel_type = parallel_type
        self.parallel_degree = parallel_degree
        self.dbyte = 2

    def get_infos(self):
        return self.m, self.n, self.k

    def get_flops(self):
        if self.type == LayerType.ALL_GATHER:
            return 0
        elif self.type == LayerType.ALL_REDUCE:
            return self.m * self.n * self.dbyte * (self.parallel_degree -1)
        else:
            assert 0, "not support comm operator type"

    def get_size(self):
        in1 = 0
        in2 = 0
        out = self.m * self.n * self.dbyte * (self.parallel_degree - 1)
        return in1, in2, out

class Transformer:

    def __init__(self, device, modelinfos, request_batch, moe_enable=True, tensor_parallel=8):
        self.tensor_parallel = tensor_parallel
        self.decoder_blocks = []
        self.moe_enable = moe_enable
        self.name = modelinfos['name']
        self.n_block = modelinfos['n_block']
        self.n_experts = modelinfos['n_experts']
        self.experts_per_token = modelinfos['experts_per_token']
        self.n_q_head = modelinfos['n_q_head']
        self.n_kv_head = modelinfos['n_kv_head']
        self.dhead = modelinfos['dhead']
        self.dim = modelinfos['dim']
        self.hdim = modelinfos['hdim']
        self.hdim_moe = modelinfos['hdim_moe']
        self.cluster_size = modelinfos['cluster_size']
        self.request_batch = request_batch
        self.device = device

    def build(self):
        batch = len(self.request_batch.request.keys())
        decoder_blocks = []
        ## QKV
        decoder_blocks.append(FC_Layer('decoder', 'qkv', LayerType.FC, "tensor_col", self.tensor_parallel, batch, self.dim, self.dhead*(self.n_q_head+2*self.n_kv_head)))

        #SpAt for gpu request-wise execution
        if self.device == DeviceType.GPU:
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Similarity, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Score_Context, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))  
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Softmax, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))
        else: # for pim all-together execution
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Similarity, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Softmax, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))
            decoder_blocks.append(Attention_Layer('decoder', 'spat', LayerType.SpAt_Score_Context, None, self.tensor_parallel, \
                self.n_q_head, self.n_kv_head, self.dhead, self.cluster_size, self.request_batch))  
                
        #Proj
        decoder_blocks.append(FC_Layer('decoder', 'proj', LayerType.FC, "tensor_row", self.tensor_parallel, batch, self.dim, self.dim))
        decoder_blocks.append(Communication('decoder', "comm", LayerType.ALL_REDUCE, self.tensor_parallel, batch, self.dim, 0))

        #Attn Norm
        decoder_blocks.append(FC_Layer('decoder', 'norm', LayerType.NORM, None, self.tensor_parallel, batch, self.dim, 0))


        #MoE FFN
        if self.moe_enable:
            #MoE Router
            decoder_blocks.append(FC_Layer('decoder', 'moe_router', LayerType.FC, "tensor", 1, batch, self.dim, self.n_experts))
            moe_activation_stat = MoE_Activation_Stat('decoder', batch, self.n_experts, self.experts_per_token).get_activated_experts()

            for expert_id, token_count in moe_activation_stat.items():
                if token_count > 0:
                    decoder_blocks.append(FC_Layer('decoder', 'moe_gate', LayerType.FC, "tensor_col", self.tensor_parallel, token_count, self.dim, self.hdim_moe))
                    decoder_blocks.append(FC_Layer('decoder', 'moe_act', LayerType.ACT, "tensor_col", self.tensor_parallel, token_count, self.dim, self.hdim_moe))
                    decoder_blocks.append(FC_Layer('decoder', 'moe_up', LayerType.FC, "tensor_col", self.tensor_parallel, token_count, self.dim, self.hdim_moe))
                    decoder_blocks.append(FC_Layer('decoder', 'moe_mat', LayerType.MATMUL, "tensor_col", self.tensor_parallel, token_count, self.hdim_moe, 1))
                    decoder_blocks.append(FC_Layer('decoder', 'moe_down', LayerType.FC, "tensor_row", self.tensor_parallel, token_count, self.hdim_moe, self.dim))
        else:
            decoder_blocks.append(FC_Layer('decoder', 'ffn_gate', LayerType.FC, "tensor_col", 1, batch, self.dim, self.n_experts))
            decoder_blocks.append(FC_Layer('decoder', 'ffn_act', LayerType.ACT, "tensor_col", self.tensor_parallel, batch, self.dim, self.hdim))
            decoder_blocks.append(FC_Layer('decoder', 'ffn_up', LayerType.FC, "tensor_col", self.tensor_parallel, batch, self.dim, self.hdim))
            decoder_blocks.append(FC_Layer('decoder', 'ffn_mat', LayerType.MATMUL, "tensor_col", self.tensor_parallel, batch, self.hdim, 1))
            decoder_blocks.append(FC_Layer('decoder', 'ffn_down', LayerType.FC, "tensor_row", self.tensor_parallel, batch, self.hdim, self.dim))

        decoder_blocks.append(Communication('decoder', "comm", LayerType.ALL_GATHER, self.tensor_parallel, batch, self.dim, 0))

        #MoE Norm
        decoder_blocks.append(FC_Layer('decoder', 'norm', LayerType.NORM, None, self.tensor_parallel, batch, self.dim, 0))


        self.decoder_blocks.append(decoder_blocks)
        return decoder_blocks