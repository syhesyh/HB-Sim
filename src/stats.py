## Define models and layer.
## Generate models


def energy_stats():
    energy_stats = {
        "qkv": 0,
        "spat_score_context": 0,
        "spat_similarity": 0,
        "spat_softmax": 0,
        "proj": 0,
        "comm": 0,
        "norm": 0,
        "moe":0,
        "ffn":0,
        "scheduling": 0,
        "promotion": 0,
        "balance": 0,
        "sum":0,
        "mem_energy": 0,
        "compute_energy": 0,
    }
    return energy_stats

def latency_stats():
    latency_stats = {
        "qkv": 0,
        "spat_score_context": 0,
        "spat_similarity": 0,
        "spat_softmax": 0,
        "proj": 0,
        "comm": 0,
        "norm": 0,
        "moe":0,
        "ffn":0,
        "scheduling": 0,
        "promotion": 0,
        "balance": 0,
        "sum":0,
    }
    return latency_stats

def tbt_stats():
    tbt = []
    return tbt

def pim_stats():
    pim_stats = {
        "pim_latency": 0,
        "pim_energy": 0,
        "sparse_pim_latency": 0,
        "sparse_pim_energy": 0,
    }
    return pim_stats