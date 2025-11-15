## Define models and layer.
## Generate models
from src.type import *
import copy
import heapq
import random
import numpy as np
from scipy.stats import zipf
import math
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




class PIM_Profile_Table():

    def __init__(self, config):
        self.num_pim_stack = config['NUM_PIM_STACK']
        self.num_pch_per_stack = config['NUM_PCH_PER_STACK']
        self.num_bg_per_pch = config['NUM_BG_PER_PCH']
        self.num_row = math.ceil(config['NUM_ROW']/48)
        self.cluster_mapping_table = {}
        self.profile_table = {}
        self.write_pointer = {}
        self.bg_stats = {}
        self.hot_set = set()
        self.cold_set = set()
        self._balance_meta_valid = False
        self._balance_meta = (0, {}, 0)  # (average, utilization, standard_deviation)
        self.full=0
        self.write_pointer["device"] = 0
        self.write_pointer["pch"] = 0
        self.write_pointer["bg"] = 0
        self.write_pointer["row"] = 0

    def build_profile_table(self):
        table = {}
        self.bg_stats = {}
        for device in range(self.num_pim_stack):
            table[device] = {}
            for pch in range(self.num_pch_per_stack):
                table[device][pch] = {}
                for bg in range(self.num_bg_per_pch):
                    table[device][pch][bg] = {}
                    for row in range(self.num_row):
                        table[device][pch][bg][row] = {
                            "request_id": None,
                            "kv_head_id": None,
                            "cluster_id": None,
                            "counts": 0,
                            "hotness_tag": 1,
                        }
                    self.bg_stats[(device, pch, bg)] = {
                        "total_counts": 0,
                        "max_row": 0 if self.num_row > 0 else None,
                        "min_row": 0 if self.num_row > 0 else None,
                    }

        self.profile_table = table
        self._mark_balance_sets_dirty()

    def _write(self, device, pch, bg, row, request_id, kv_head_id, cluster_id, counts=1):
        old_counts = self.profile_table[device][pch][bg][row]["counts"]
        self.profile_table[device][pch][bg][row] = {
            "request_id": request_id,
            "kv_head_id": kv_head_id,
            "cluster_id": cluster_id,
            "counts": counts,
            "hotness_tag": counts//62.5
        }
        self._update_bg_stats(device, pch, bg, row, counts, old_counts)


    def _get_bg_stats(self, device, pch, bg):
        return self.bg_stats[(device, pch, bg)]


    def _get_row_counts(self, device, pch, bg, row):
        return self.profile_table[device][pch][bg][row]["counts"]


    def _find_extreme_row(self, device, pch, bg, find_max):
        target_row = None
        target_count = None
        for row in range(self.num_row):
            counts = self._get_row_counts(device, pch, bg, row)
            if target_count is None:
                target_count = counts
                target_row = row
                continue
            if find_max:
                if counts > target_count or (counts == target_count and row < target_row):
                    target_count = counts
                    target_row = row
            # find min row
            else:
                if counts < target_count or (counts == target_count and row < target_row):
                    target_count = counts
                    target_row = row
        return target_row


    def _update_bg_stats(self, device, pch, bg, row, new_counts, old_counts):
        stats = self._get_bg_stats(device, pch, bg)
        stats["total_counts"] += new_counts - old_counts

        max_row = stats["max_row"]
        if max_row is None or new_counts > self._get_row_counts(device, pch, bg, max_row) or (
            new_counts == self._get_row_counts(device, pch, bg, max_row) and row < max_row
        ):
            stats["max_row"] = row
        elif max_row == row and new_counts < old_counts:
            stats["max_row"] = self._find_extreme_row(device, pch, bg, find_max=True)

        min_row = stats["min_row"]
        if min_row is None or new_counts < self._get_row_counts(device, pch, bg, min_row) or (
            new_counts == self._get_row_counts(device, pch, bg, min_row) and row < min_row
        ):
            stats["min_row"] = row
        elif min_row == row and new_counts > old_counts:
            stats["min_row"] = self._find_extreme_row(device, pch, bg, find_max=False)

        self._mark_balance_sets_dirty()


    def _recompute_bg_stats(self, device, pch, bg):
        stats = self._get_bg_stats(device, pch, bg)
        total_counts = 0
        max_row = None
        min_row = None
        max_count = None
        min_count = None

        for row in range(self.num_row):
            counts = self._get_row_counts(device, pch, bg, row)
            total_counts += counts
            if max_count is None or counts > max_count or (counts == max_count and (max_row is None or row < max_row)):
                max_count = counts
                max_row = row
            if min_count is None or counts < min_count or (counts == min_count and (min_row is None or row < min_row)):
                min_count = counts
                min_row = row

        stats["total_counts"] = total_counts
        stats["max_row"] = max_row
        stats["min_row"] = min_row
        self._mark_balance_sets_dirty()


    def _mark_balance_sets_dirty(self):
        self._balance_meta_valid = False


    def _recalculate_balance_sets(self):
        utilization = {key: stats["total_counts"] for key, stats in self.bg_stats.items()}
        if utilization:
            average = sum(utilization.values()) / len(utilization)
        else:
            average = 0
        self.hot_set = {key for key, total in utilization.items() if total > average}
        self.cold_set = {key for key, total in utilization.items() if total < average}
        
        # 计算标准差：sqrt(sum((x_i - mean)^2) / n)
        if len(utilization) > 0:
            standard_deviation = np.std(list(utilization.values()), ddof=0)
        else:
            standard_deviation = 0
        
        self._balance_meta = (average, utilization, standard_deviation)
        self._balance_meta_valid = True


    def _get_balance_meta(self):
        if not self._balance_meta_valid:
            self._recalculate_balance_sets()
        average, utilization, total_variance = self._balance_meta
        return average, utilization, total_variance


    def _update_cluster_mapping_for_entry(self, entry, stack, pch, bg, row):
        request_id = entry.get("request_id")
        if request_id is None:
            return
        key = (request_id, entry["kv_head_id"], entry["cluster_id"])
        self.cluster_mapping_table[key] = (stack, pch, bg, row)


    def _remove_cluster_mapping_for_entry(self, entry):
        request_id = entry.get("request_id")
        if request_id is None:
            return
        key = (request_id, entry["kv_head_id"], entry["cluster_id"])
        self.cluster_mapping_table.pop(key, None)


    def _swap_entries(self, hot_key, hot_row, cold_key, cold_row):
        hot_stack, hot_pch, hot_bg = hot_key
        cold_stack, cold_pch, cold_bg = cold_key

        hot_entry = self.profile_table[hot_stack][hot_pch][hot_bg][hot_row].copy()
        cold_entry = self.profile_table[cold_stack][cold_pch][cold_bg][cold_row].copy()

        self.profile_table[hot_stack][hot_pch][hot_bg][hot_row] = cold_entry
        self.profile_table[cold_stack][cold_pch][cold_bg][cold_row] = hot_entry

        self._update_cluster_mapping_for_entry(
            self.profile_table[hot_stack][hot_pch][hot_bg][hot_row],
            hot_stack,
            hot_pch,
            hot_bg,
            hot_row,
        )
        self._update_cluster_mapping_for_entry(
            self.profile_table[cold_stack][cold_pch][cold_bg][cold_row],
            cold_stack,
            cold_pch,
            cold_bg,
            cold_row,
        )

        self._recompute_bg_stats(hot_stack, hot_pch, hot_bg)
        self._recompute_bg_stats(cold_stack, cold_pch, cold_bg)
        self._mark_balance_sets_dirty()


    def _wp_update(self):
        temp_wp = self.write_pointer
        print(f"temp_wp: {temp_wp}")
        if self.full == 1:
            return
        n_scan=0
        while(True):
            if temp_wp["device"] < self.num_pim_stack-1:
                temp_wp["device"] += 1
            else:
                temp_wp["device"] = 0
                if temp_wp["pch"] < self.num_pch_per_stack-1:
                    temp_wp["pch"] += 1
                else:
                    temp_wp["pch"] = 0
                    if temp_wp["bg"] < self.num_bg_per_pch-1:
                        temp_wp["bg"] += 1
                    else:
                        temp_wp["bg"] = 0
                        if temp_wp["row"] < self.num_row-1:
                            temp_wp["row"] += 1
                        else:
                            temp_wp["row"] = 0

            print(f"temp_wp: {temp_wp}")                        
            if self.profile_table[temp_wp["device"]][temp_wp["pch"]][temp_wp["bg"]][temp_wp["row"]]["request_id"] is None:
                self.write_pointer = temp_wp
                n_scan += 1
                if n_scan == self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch * self.num_row:
                    self.full = 1
                    return
            if n_scan == self.num_pim_stack * self.num_pch_per_stack * self.num_bg_per_pch * self.num_row:
                self.full = 1
                return


    def _cluster_mapping_update(self, request_id, kv_head_id, cluster_id, stack, pch, bg, row):
        self.cluster_mapping_table[(request_id, kv_head_id, cluster_id)] = (stack, pch, bg, row)


    def _reset_entry(self, stack, pch, bg, row):
        entry = {
            "request_id": None,
            "kv_head_id": None,
            "cluster_id": None,
            "counts": 0,
            "hotness_tag": 1,
        }
        old_counts = self.profile_table[stack][pch][bg][row]["counts"]
        self.profile_table[stack][pch][bg][row] = entry
        self._update_bg_stats(stack, pch, bg, row, entry["counts"], old_counts)


    def append(self, request_id, kv_head_id, cluster_id):

        if self.full == 0:
            stack = self.write_pointer["stack"]
            pch = self.write_pointer["pch"]
            bg = self.write_pointer["bg"]
            row = self.write_pointer["row"]
            print(f"Successfully write to PIM Profile Table, request_id: {request_id}, kv_head_id: {kv_head_id}, cluster_id: {cluster_id}")
            self._write(stack, pch, bg, row, request_id, kv_head_id, cluster_id)
            self._cluster_mapping_update(request_id, kv_head_id, cluster_id, stack, pch, bg, row)
            self._wp_update()
            self._recalculate_balance_sets()
        else:
            print(f"PIM Profile Table is full")



    def update(self, request_id, kv_head_id, cluster_id):
        key = (request_id, kv_head_id, cluster_id)
        location = self.cluster_mapping_table.get(key)
        if location is None:
            return

        stack, pch, bg, row = location
        entry = self.profile_table[stack][pch][bg][row]
        if (
            entry["request_id"] != request_id
            or entry["kv_head_id"] != kv_head_id
            or entry["cluster_id"] != cluster_id
        ):
            return

        old_counts = entry["counts"]
        entry["counts"] += 1
        entry["hotness_tag"] = entry["counts"] // 62.5
        self._update_bg_stats(stack, pch, bg, row, entry["counts"], old_counts)
        self._recalculate_balance_sets()


    def request_exit(self, request_id):
        keys_to_remove = [
            key for key in list(self.cluster_mapping_table.keys()) if key[0] == request_id
        ]

        if keys_to_remove:
            for key in keys_to_remove:
                stack, pch, bg, row = self.cluster_mapping_table.pop(key)
                self._reset_entry(stack, pch, bg, row)
            self._recalculate_balance_sets()
            return

    
    def leakage_average(self):
    # 遍历所有device, pch, bg, row位置
        for device in range(self.num_pim_stack):
            for pch in range(self.num_pch_per_stack):
                for bg in range(self.num_bg_per_pch):
                    for row in range(self.num_row):
                        entry = self.profile_table[device][pch][bg][row]
                        entry["counts"] = entry["counts"] // 2
                        entry["hotness_tag"] = entry["counts"] // 62.5
                    self._recompute_bg_stats(device, pch, bg)


    def greedy_balance_load(self, error, stabilization_iteration=64, stabilization_factor=0.1):
        if error < 0:
            raise ValueError("error must be non-negative")

        average, utilization, _ = self._get_balance_meta()

        # 标准差阈值：error 作为相对误差，threshold = average * error
        std_threshold = average * error
        max_iterations = len(utilization) * max(self.num_row, 1)
        std_history = []

        def classify_sets(current_utilization):
            hot = [key for key, total in current_utilization.items() if total > average]
            cold = [key for key, total in current_utilization.items() if total < average]
            return hot, cold

        def compute_standard_deviation(current_utilization):
            """计算标准差：sqrt(sum((x_i - mean)^2) / n)"""
            if len(current_utilization) == 0:
                return 0
            return np.std(list(current_utilization.values()), ddof=0)

        hot_set, cold_set = classify_sets(utilization)
        initial_std = compute_standard_deviation(utilization)

        if initial_std <= std_threshold:
            return initial_std, initial_std, [], 0

        final_std = initial_std
        swaps = 0

        for _ in range(max_iterations):

            hot_key = max(hot_set, key=lambda key: utilization[key] - average)
            cold_key = max(cold_set, key=lambda key: average - utilization[key])

            hot_row = self.bg_stats[hot_key]["max_row"]
            cold_row = self.bg_stats[cold_key]["min_row"]

            hot_counts = self._get_row_counts(*hot_key, hot_row)
            cold_counts = self._get_row_counts(*cold_key, cold_row)

            if hot_counts <= cold_counts:
                swaps = swaps - 1 if swaps > 0 else 0
                break

            self._swap_entries(hot_key, hot_row, cold_key, cold_row)
            utilization[hot_key] = self.bg_stats[hot_key]["total_counts"]
            utilization[cold_key] = self.bg_stats[cold_key]["total_counts"]
            swaps += 1

            # hot_set, cold_set = classify_sets(utilization)
            final_std = compute_standard_deviation(utilization)
            std_history.append(final_std)

            if final_std <= std_threshold:
                break

            if len(std_history) >= stabilization_iteration:
                recent = std_history[-stabilization_iteration:]
                if max(recent) - min(recent) <= sum(recent) / len(recent) * stabilization_factor:
                    break


        self._recalculate_balance_sets()
        return initial_std, final_std, std_history, swaps




class HBF_Track_Table():
    def __init__(self, n_entry):
        self.n_entry = n_entry
        self.build_hbf_track_table()

    def build_hbf_track_table(self):
        self.entries = {}
        self.min_heap = []
        self.entry_tokens = {}
        self._heap_counter = 0
        self.table = self.entries
        return self.entries

    def _push_heap(self, key, counts):
        self._heap_counter += 1
        token = self._heap_counter
        self.entry_tokens[key] = token
        heapq.heappush(self.min_heap, (counts, token, key))

    def _evict_min(self):
        while self.min_heap:
            counts, token, key = heapq.heappop(self.min_heap)
            if self.entry_tokens.get(key) != token:
                continue
            entry = self.entries.pop(key, None)
            if entry is None:
                self.entry_tokens.pop(key, None)
                continue
            self.entry_tokens.pop(key, None)
            return counts, entry
        return 0, None

    def _insert_entry(self, request_id, kv_head_id, cluster_id, counts):
        key = (request_id, kv_head_id, cluster_id)
        entry = {
            "request_id": request_id,
            "kv_head_id": kv_head_id,
            "cluster_id": cluster_id,
            "counts": counts,
            "hotness_tag": counts//62.5
        }
        self.entries[key] = entry
        self._push_heap(key, counts)
        return entry

    def update(self, request_id, kv_head_id, cluster_id, weight=1):
        if weight <= 0:
            return None

        key = (request_id, kv_head_id, cluster_id)
        if key in self.entries:
            entry = self.entries[key]
            entry["counts"] += weight
            entry["hotness_tag"] = entry["counts"] // 62.5
            self._push_heap(key, entry["counts"])
            return entry

        if len(self.entries) < self.n_entry:
            return self._insert_entry(request_id, kv_head_id, cluster_id, weight)

        min_count, _ = self._evict_min()
        counts = min_count + weight if min_count else weight
        return self._insert_entry(request_id, kv_head_id, cluster_id, counts)

    def get(self, request_id, kv_head_id, cluster_id):
        key = (request_id, kv_head_id, cluster_id)
        return self.entries.get(key)

    def top_k(self, k=None):
        items = sorted(
            self.entries.values(), key=lambda entry: entry["counts"], reverse=True
        )
        if k is not None:
            items = items[:k]
        return copy.deepcopy(items)

    def remove(self, request_id, kv_head_id, cluster_id):
        key = (request_id, kv_head_id, cluster_id)
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        self.entry_tokens.pop(key, None)
        return entry

    def leakage_average(self):
        if not self.entries:
            return

        for entry in self.entries.values():
            entry["counts"] = entry["counts"] // 2
            entry["hotness_tag"] = entry["counts"] // 62.5

    def clear(self):
        self.build_hbf_track_table()


def promotion(hbf_table, pim_table, k=None):

    if hbf_table is None or pim_table is None:
        raise ValueError("hbf_table and pim_table must be provided")

    hbf_candidates = hbf_table.top_k(k)

    average, utilization, _ = pim_table._get_balance_meta()

    # 先按照冷 bg 与平均值的差距从大到小排序，差距越大优先级越高；再把其余 bg 也按总计数从低到高排列，拼成优先队列 key_order。
    cold_keys = sorted(
        pim_table.cold_set,
        key=lambda key: abs(average - utilization[key]),
        reverse=True,
    )
    remaining_keys = sorted(
        [key for key in utilization.keys() if key not in pim_table.cold_set],
        key=lambda key: utilization[key],
    )
    key_order = cold_keys + remaining_keys

    promotions = []
    n_promotions = 0
    key_index = 0  # 当前在 key_order 中的索引
    
    for entry in hbf_candidates:
        # 尝试找到一个合适的 PIM 条目进行替换
        found = False
        attempts = 0
        max_attempts = len(key_order)  # 最多尝试所有 bank 一次
        
        while attempts < max_attempts and not found:
            # 循环遍历 key_order，允许重复访问同一个 bank
            key = key_order[key_index % len(key_order)]
            key_index += 1
            attempts += 1
            
            # 获取该 bank 的 min_row
            row = pim_table.bg_stats[key]["min_row"]
            if row is None:
                continue
            
            pim_counts = pim_table._get_row_counts(*key, row)
            
            # 如果 HBF 条目的 counts 大于 PIM 条目的 counts，进行替换
            if entry["counts"] > pim_counts:
                device, pch, bg = key
                old_entry = copy.deepcopy(pim_table.profile_table[device][pch][bg][row])
                pim_table._remove_cluster_mapping_for_entry(old_entry)
                pim_table._write(
                    device,
                    pch,
                    bg,
                    row,
                    entry["request_id"],
                    entry["kv_head_id"],
                    entry["cluster_id"],
                    entry["counts"],
                )
                pim_table._cluster_mapping_update(
                    entry["request_id"],
                    entry["kv_head_id"],
                    entry["cluster_id"],
                    device,
                    pch,
                    bg,
                    row,
                )
                
                # 重新计算该 bank 的统计信息，更新 min_row
                pim_table._recompute_bg_stats(device, pch, bg)
                
                # 在 HBF 中用被替换的 PIM 条目替换晋升的条目
                hbf_table.remove(entry["request_id"], entry["kv_head_id"], entry["cluster_id"])
                if old_entry["request_id"] is not None and old_entry["counts"] > 0:
                    hbf_table._insert_entry(
                        old_entry["request_id"],
                        old_entry["kv_head_id"],
                        old_entry["cluster_id"],
                        old_entry["counts"],
                    )

                promotions.append(
                    {
                        "hbf_entry": entry,
                        "replaced_entry": old_entry,
                        "location": {"device": device, "pch": pch, "bg": bg, "row": row},
                    }
                )
                n_promotions += 1
                found = True
                if n_promotions >= k:
                    break
            else:
                # 如果当前 bank 的 min_row 不够小，继续尝试下一个 bank
                continue
        
        # 如果找不到合适的替换目标，停止处理剩余的 HBF 候选
        if not found:
            break

    pim_table._recalculate_balance_sets()
    return promotions, n_promotions