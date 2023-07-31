#!/usr/bin/python3
import os
import os.path as osp
import sys
import random
import profile_cti_exec as profiler


random.seed(1)
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")

sti_data_stat_dirpath = osp.join(snowcat_storage_dirpath, "sti-data", "stat")
assert osp.exists(sti_data_stat_dirpath) is True

profiled_sti_id_list = []
for f in os.scandir(sti_data_stat_dirpath):
    profiled_sti_id_list.append(int(f.name))

def get_random_sti_pair(sti_id_list, max_num_cti=None):
    """Randomly pair two stis to generate a cti"""
    random.shuffle(sti_id_list)
    sti_id_list_a = sti_id_list
    random.shuffle(sti_id_list)
    sti_id_list_b = sti_id_list
    cti_id_list = []
    for sti_id_a in sti_id_list_a:
        for sti_id_b in sti_id_list_b:
            cti_id_list.append([sti_id_a, sti_id_b])
            if max_num_cti is not None:
                if len(cti_id_list) == max_num_cti:
                    break
        if max_num_cti is not None:
            if len(cti_id_list) == max_num_cti:
                break
    return cti_id_list

if len(sys.argv) == 1:
    # TODO: provide a simple ETA
    print("[Waring] No limits on number of CTIs to profile are found.")
    print(f"Will profile {len(sti_id_list) * len(sti_id_list)} CTIs.")
    import time
    time.sleep(5)
    cti_id_list = get_random_sti_pair(profiled_sti_id_list)
else:
    max_num_cti = int(sys.argv[1])
    cti_id_list = get_random_sti_pair(profiled_sti_id_list, max_num_cti)
print(f"Will profile {len(cti_id_list)} CTIs.")

for cti_id in cti_id_list:
    profiler.data_collector(cti_id)
