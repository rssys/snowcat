import os
import os.path as osp
import sys
import random
import pickle
import pandas as pd


def fetch_pos_block_addr_set(ct_inference_filepath):
    pos_block_filepath = ct_inference_filepath + ".pos_block_cache"
    with open(pos_block_filepath, "rb") as tmp_file:
        return pickle.load(tmp_file)

def get_cti_id(ct_inference_filepath):
    cti_id = ct_inference_filepath.split("/")[-2]
    int(cti_id.split("_")[0])
    int(cti_id.split("_")[1])
    return cti_id

def emulate_one_cti(inference_data_dirpath, repeat=1000):
    cti_id_list_filepath = osp.join(inference_data_dirpath, "cti_id_list")
    cti_id_set = set([])
    buggy_cti_id_set = set([])
    with open(cti_id_list_filepath, "r") as tmp_file:
        for line in tmp_file:
            line_split = line.strip().split(" ")
            cti_id_set.add(line_split[0])
            # Only the cti that is labeled as True can expose the bug
            if line_split[1] == "True":
                buggy_cti_id_set.add(line_split[0])
    print(inference_data_dirpath)

    ct_inference_filepath_list = []
    for tmp_f in os.scandir(inference_data_dirpath):
        if tmp_f.is_dir() is False:
            continue
        try:
            int(tmp_f.name.split("_")[0])
            int(tmp_f.name.split("_")[1])
        except:
            continue
        for tmp_tmp_f in os.scandir(tmp_f.path):
            if tmp_tmp_f.name.endswith(".npy") is True:
                ct_inference_filepath_list.append(tmp_tmp_f.path)

    bug_find_prob_by_method = {}
    avg_num_selected_cti_by_method = {}
    strategy_list = ["S2"]
    for strategy in strategy_list:
        # use SB to select cti
        num_bug_found_run = 0
        num_selected_cti = 0
        for run_idx in range(repeat):
            unique_set = set([])
            predicted_cti_set = set([])
            selected_cti_set = set([])
            random.seed(run_idx)
            random.shuffle(ct_inference_filepath_list)
            for ct_inference_filepath in ct_inference_filepath_list:
                pos_block_set = fetch_pos_block_addr_set(ct_inference_filepath)
                cti_id = get_cti_id(ct_inference_filepath)
                predicted_cti_set.add(cti_id)
                if strategy == "S1":
                    pos_block_set = frozenset(pos_block_set)
                    if pos_block_set in unique_set:
                        continue
                    unique_set.add(pos_block_set)
                    selected_cti_set.add(cti_id)
                else:
                    assert strategy == "S2"
                    if len(pos_block_set - unique_set) == 0:
                        continue
                    unique_set |= pos_block_set
                    selected_cti_set.add(cti_id)
            selected_cti_set |= (cti_id_set - predicted_cti_set)
            if len(selected_cti_set & buggy_cti_id_set) > 0:
                num_bug_found_run += 1
            num_selected_cti += len(selected_cti_set)
        bug_find_prob_by_method[f"SB-PIC-{strategy}"] = num_bug_found_run / repeat
        avg_num_selected_cti_by_method[f"SB-PIC-{strategy}"] = int(num_selected_cti / repeat)
        avg_num_selected_cti = int(num_selected_cti / repeat)

        # emulate SB-RAND
        num_bug_found_run = 0
        num_selected_cti = 0
        for run_idx in range(repeat):
            random.seed(run_idx)
            cti_id_list = list(cti_id_set)
            random.shuffle(cti_id_list)
            selected_cti_set = set(cti_id_list[ : avg_num_selected_cti])
            if len(selected_cti_set & buggy_cti_id_set) > 0:
                num_bug_found_run += 1
            num_selected_cti += len(selected_cti_set)
        bug_find_prob_by_method[f"SB-RAND-{strategy}"] = num_bug_found_run / repeat
        avg_num_selected_cti_by_method[f"SB-RAND-{strategy}"] = int(num_selected_cti / repeat)
    print("Bug finding probability:")
    for method in bug_find_prob_by_method:
        prob = bug_find_prob_by_method[method]
        print(f"Method: {method} Probability: {prob}")
    print("Average number of selected ctis:")
    for method in avg_num_selected_cti_by_method:
        avg_num_selected_cti = avg_num_selected_cti_by_method[method]
        print(f"Method: {method} #-selected-ctis-avg: {avg_num_selected_cti}")

inference_data_dirpath = sys.argv[1]
emulate_one_cti(inference_data_dirpath)
