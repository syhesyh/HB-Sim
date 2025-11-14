## Define models and layer.
## Generate models


def energy_stats():
    energy_stats = {
        "gpu_spat_others": 0,
        "gpu_spat_score_context": 0,
        "pim_spat_score_context": 0,
        "comm": 0,
        "fc": 0,
        "scheduling": 0,
        "promotion": 0,
        "balance": 0,
    }
    return energy_stats

def latency_stats():
    latency_stats = {
        "gpu_spat_others": 0,
        "gpu_spat_score_context": 0,
        "pim_spat_score_context": 0,
        "comm": 0,
        "fc": 0,
        "scheduling": 0,
        "promotion": 0,
        "balance": 0,
    }
    return latency_stats