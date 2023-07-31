import os
import os.path as osp
import sys
import multiprocessing
import traceback
from torch_geometric.loader import DataLoader
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
sys.path.append(osp.join(main_home, "learning", "dataset"))
import PICDataset
sys.path.append(osp.join(main_home, "learning", "dataset", "generator"))
import testinput

snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
dataset_dirpath = osp.join(snowcat_storage_dirpath, "cti-data", "dataset")
if osp.exists(dataset_dirpath) is False:
    print(f"No dataset exists in {dataset_dirpath}?")
    exit(1)

kernel_info_dirpath = os.environ.get('KERNEL_INFO_DIR')
if kernel_info_dirpath is None:
    print("Please source choose_kernel.sh")
    exit(1)
KERNEL_INFO_DIRPATH = kernel_info_dirpath

INS_TO_BLOCK_DICT = {}
def init_ins_to_block_dict():
    """Create a dict that maps the instruction address to the block address."""
    global INS_TO_BLOCK_DICT
    global KERNEL_INFO_DIRPATH
    block_range_list = []
    vmlinux_map_filepath = osp.join(KERNEL_INFO_DIRPATH, "vmlinux.map")
    with open(vmlinux_map_filepath, "r") as vmlinux_map_file:
        for line in vmlinux_map_file:
            ins_addr = int(line.strip()[ : 10], 16)
            INS_TO_BLOCK_DICT[ins_addr] = -1
    block_info_filepath = osp.join(KERNEL_INFO_DIRPATH, "block-info")
    with open(block_info_filepath, "r") as block_info_file:
        for line in block_info_file:
            line_split = line.strip().split(" ")
            block_start_addr = int(line_split[0], 16)
            block_end_addr = int(line_split[1], 16)
            block_range = tuple([block_start_addr, block_end_addr])
            block_range_list.append(block_range)
            for ins_addr in range(block_start_addr, block_end_addr):
                if ins_addr not in INS_TO_BLOCK_DICT:
                    continue
                INS_TO_BLOCK_DICT[ins_addr] = block_start_addr
    num_block = len(block_range_list)
    num_ins = len(INS_TO_BLOCK_DICT.keys())
    print(f"Mapped {num_ins} instructions to {num_block} blocks")


init_ins_to_block_dict()

ct_list = []
for tmp_f in os.scandir(dataset_dirpath):
    if tmp_f.is_dir() is False:
        continue
    cti_data_dirpath = osp.join(tmp_f.path, "schedule_info")
    for f in os.scandir(cti_data_dirpath):
        # Format of the schedule_info is
        # {STI_A_ID}_{STI_B_ID}_{INIT_CPU}_{CPU0_SWITCH_POINT}_{CPU1_SWITCH_POINT}
        schedule_info = f.name.split("_")
        sti_a = testinput.STI(schedule_info[0], "cpu0")
        sti_b = testinput.STI(schedule_info[1], "cpu1")
        cti = testinput.CTI(sti_a, sti_b)
        if schedule_info[2] == "0":
            init_cpu = "cpu0"
        else:
            assert schedule_info[2] == "1"
            init_cpu = "cpu1"
        cpu0_switch_point = int(schedule_info[3], 16)
        cpu1_switch_point = int(schedule_info[4], 16)
        try:
            cpu0_switch_point = INS_TO_BLOCK_DICT[cpu0_switch_point]
            cpu1_switch_point = INS_TO_BLOCK_DICT[cpu1_switch_point]
        except:
            continue
        schedule = testinput.SCHEDULE(init_cpu, cpu0_switch_point, \
                cpu1_switch_point)
        ct_coverage_data_dirpath = osp.join(cti_data_dirpath, f.name)
        coverage_by_cpu = {}
        cpu0_coverage_filepath = \
                osp.join(ct_coverage_data_dirpath, "cpu0-coverage")
        cpu1_coverage_filepath = \
                osp.join(ct_coverage_data_dirpath, "cpu1-coverage")
        if osp.exists(cpu0_coverage_filepath) is False or \
                osp.exists(cpu1_coverage_filepath) is False:
                    continue
        coverage_by_cpu["cpu0"] = cpu0_coverage_filepath
        coverage_by_cpu["cpu1"] = cpu1_coverage_filepath
        ct = testinput.CT(cti, schedule, coverage_by_cpu)
        #cti_id_set.add(f"{schedule_info[0]}_{schedule_info[1]}")
        # We only need the info of one schedule per CTI
        ct_list.append(ct)
        break


sti_data_dirpath = osp.join(snowcat_storage_dirpath, "sti-data")
run_dirpath = "./test-tmp"
os.makedirs(run_dirpath, exist_ok=True)

config = PICDataset.PICDatasetConfig(\
        sti_data_dirpath, \
        8, run_dirpath)
dataset = PICDataset.PICDataset(ct_list, config)
num_worker = multiprocessing.cpu_count()

def extract_cti_id_from_ct_id(ct_id_tuple):
    assert isinstance(ct_id_tuple, tuple)
    assert len(ct_id_tuple) == 1
    ct_id = ct_id_tuple[0]
    ct_id_split = ct_id.split("_")
    sti_a_id = ct_id_split[0]
    sti_b_id = ct_id_split[1]
    return f"{sti_a_id}_{sti_b_id}"

loader = DataLoader(dataset, batch_size=1, num_workers=num_worker)
cti_set = set([])
toolarge_cti_set = set([])
for graph, ct_id in loader:
    stat_dict = {}
    try:
        #ur_node_indices = (graph.pos == 1).nonzero()[ : , 0]
        num_node = graph.x.shape[0]
        """In our experiments, we observed that a too large graph
        (e.g., a graph that has >25000 nodes) could cause GPU OOM,
        which then crashes the training.
        Thus, we want to avoid large graphs in the training dataset.
        Note this is not 'cheating' because the large graphs still go to
        the validation/test dataset. We probably would have a lower validation
        performance by doing this.
        """
        cti_id = extract_cti_id_from_ct_id(ct_id)
        if num_node > 25000:
            toolarge_cti_set.add(cti_id)
        cti_set.add(cti_id)
    except:
        #traceback.print_exc()
        continue

"""
We split the dataset into train, valiation and test here.
"""
from datetime import datetime
import random

train_cti_set = set([])
val_cti_set = set([])
test_cti_set = set([])
random.seed(1)
cti_list = list(cti_set)
random.shuffle(cti_list)
num_train_cti = int(len(cti_list) * 0.8)
num_val_cti = int(len(cti_list) * 0.1)
train_cti_set = set(cti_list[ : num_train_cti])
val_cti_set = set(cti_list[num_train_cti : num_train_cti + num_val_cti])
test_cti_set = set(cti_list[num_train_cti + num_val_cti : ])
needmove_cti_set = train_cti_set & toolarge_cti_set
train_cti_set = train_cti_set - toolarge_cti_set
test_cti_set = test_cti_set | needmove_cti_set

time_now = datetime.now()
timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
dataset_split_dirpath = osp.join("./dataset_split", f"split-{timestamp}")
os.makedirs(dataset_split_dirpath, exist_ok=True)
with open(osp.join(dataset_split_dirpath, "train_cti_list"), "w") as tmp_file:
    for cti_id in train_cti_set:
        print(cti_id, file=tmp_file)
with open(osp.join(dataset_split_dirpath, "val_cti_list"), "w") as tmp_file:
    for cti_id in val_cti_set:
        print(cti_id, file=tmp_file)
with open(osp.join(dataset_split_dirpath, "test_cti_list"), "w") as tmp_file:
    for cti_id in test_cti_set:
        print(cti_id, file=tmp_file)


import configparser
template_config_filepath = "./train-config-template.ini"
config = configparser.ConfigParser()
config.read(template_config_filepath)
config["dataset"]["dataset_dirpath"] = dataset_dirpath
config["dataset"]["dataset_split_dirpath"] = dataset_split_dirpath
config["dataset"]["sti_data_dirpath"] = sti_data_dirpath

train_config_filepath = f"./train-config-{timestamp}.ini"
with open(train_config_filepath, "w") as tmp_file:
    config.write(tmp_file)

template_config_filepath = "./predict-config-template.ini"
config = configparser.ConfigParser()
config.read(template_config_filepath)
config["dataset"]["dataset_dirpath"] = dataset_dirpath
config["dataset"]["sti_data_dirpath"] = sti_data_dirpath
config["inference"]["cti_list_filepath"] = osp.join(dataset_split_dirpath, "test_cti_list")
predict_config_filepath = f"./predict-config-{timestamp}.ini"
with open(predict_config_filepath, "w") as tmp_file:
    config.write(tmp_file)

print(f"the new dataset split is stored in {dataset_split_dirpath}")
print(f"the new training config is stored in {train_config_filepath}")
print(f"the new inference/predict config is stored in {predict_config_filepath}")
