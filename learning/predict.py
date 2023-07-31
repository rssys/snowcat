import os
import os.path as osp
import sys
import configparser
import datetime
import numpy
import torch
import traceback
from torch_geometric.loader import DataLoader
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
#import custom modules
import model.model as Model
from dataset import PICDataset


config_filepath = sys.argv[1]
if osp.exists(config_filepath) is not True:
    print(f"config file {config_filepath} does not exist")
    exit(-1)

# Configure the inference based on the user config
config = configparser.ConfigParser()
config.read(config_filepath)
DATASET_DIRPATH = config["dataset"]["dataset_dirpath"]
SHORTCUT_EDGE_HOP = int(config["dataset"]["shortcut_edge_hop"])
if config["dataset"]["num_ct_per_cti"].lower() == "none":
    SAMPLE_SIZE = None
else:
    SAMPLE_SIZE = int(config["dataset"]["num_ct_per_cti"])
STI_DATA_DIRPATH = config["dataset"]["sti_data_dirpath"]

PRETRAINED_BERT_FILEPATH = config["model"]["pretrained_bert_ckpt_filepath"]
assert osp.exists(PRETRAINED_BERT_FILEPATH) is True

if len(sys.argv) == 3:
    CHECKPOINT_FILEPATH = sys.argv[2]
else:
    CHECKPOINT_FILEPATH = config["model"]["checkpoint_filepath"]
if osp.exists(CHECKPOINT_FILEPATH) is False:
    print(f"Checkpoint {CHECKPOINT_FILEPATH} does not exist")
    exit(0)

BATCH_SIZE = int(config["inference"]["batch_size"])
CTI_LIST_FILEPATH = config["inference"]["cti_list_filepath"]
if osp.exists(CTI_LIST_FILEPATH) is False:
    print("Cti list file {CTI_LIST_FILEPATH} does not exist")
    exit(0)
USE_CPU = config["inference"].getboolean("use_cpu")
USE_AMP = config["inference"].getboolean("use_amp")
if USE_AMP is True:
    print("Changing use_amp to False because it is unnecessary")
    USE_AMP = False


def create_backup(run_dirpath, test_ct_list):
    """Create a backup for reproducing results."""
    backup_dirpath = osp.join(run_dirpath, "backup")
    os.makedirs(backup_dirpath, exist_ok=True)
    cmd = f"cp {config_filepath} {backup_dirpath}/"
    os.system(cmd)
    cmd = f"cp predict.py {backup_dirpath}/"
    os.system(cmd)
    cmd = f"cp dataset/PICDataset.py {backup_dirpath}"
    os.system(cmd)
    with open(osp.join(backup_dirpath, "test_ct_list"), "w") as tmp_file:
        for ct in test_ct_list:
            print(ct.id, file=tmp_file)


def save_pred_true(result_dirpath, ct_id, ct_pred_true):
    """Save y_true and y_pred to a local file."""
    try:
        ct_pred_true = ct_pred_true.numpy()
        ct_id_fields = ct_id.split("_")
        cti_id = "_".join([ct_id_fields[0], ct_id_fields[1]])
        cti_result_dirpath = osp.join(result_dirpath, cti_id)
        # there might be several processes creating the dir concurrently,
        # so we need to check the existance first.
        if osp.exists(cti_result_dirpath) is False:
            os.makedirs(cti_result_dirpath, exist_ok=True)
        result_filepath = osp.join(cti_result_dirpath, ct_id)
        numpy.save(result_filepath, ct_pred_true, allow_pickle=True)
        print(f"Saved the inference of CT {ct_id}")
    except:
        traceback.print_exc()
        log_filename = f"save_pred_true-error-{os.getpid()}"
        log_filepath = osp.join(result_dirpath, log_filename)
        with open(log_filepath, "a") as f:
            traceback.print_exc(file=f)


def inference(rank, world_size, run_dirpath, dataset):
    """Make inference on each CT graph and save the inference to disk."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist_protocol = "gloo" if USE_CPU is True else "nccl"
    dist.init_process_group(dist_protocol, rank=rank, world_size=world_size)

    # One can increase/decrease the value of `num_workers` to 
    # control the cpu pressure. We used the value 48 on our machines.
    graph_sampler = DistributedSampler(dataset, num_replicas=world_size, \
            rank=rank, drop_last=False)
    graph_loader = DataLoader(dataset, batch_size=BATCH_SIZE, \
            sampler=graph_sampler, pin_memory=True,
            num_workers=12, prefetch_factor=64)

    if USE_CPU is False:
        device = torch.device('cuda:%d' % rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    torch.manual_seed(12345)

    # Load the model checkpoint
    model = Model.create_model(config["model"])
    model.to(device)
    assert osp.exists(CHECKPOINT_FILEPATH) is True
    if USE_CPU is False:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model_checkpoint = torch.load(CHECKPOINT_FILEPATH, map_location=map_location)
    else:
        model_checkpoint = torch.load(CHECKPOINT_FILEPATH, map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint["model"])

    model.eval()
    num_worker = BATCH_SIZE * 4 # number of workers to do the IO
    io_worker_pool = multiprocessing.Pool(num_worker)
    for batched_graph, ct_id_list in graph_loader:
        with torch.no_grad():
            batched_graph.to(device)
            block_token = batched_graph.x
            edge = batched_graph.edge_index
            edge_type = batched_graph.edge_attr
            block_batch_idx = batched_graph.batch

            batched_y_true = batched_graph.y
            batched_y_pred = model(block_token, edge, edge_type)
            batched_y_pred = torch.sigmoid(batched_y_pred)

            num_graphs = len(ct_id_list)
            for batch_idx in range(num_graphs):
                this_ct_block_idx = (block_batch_idx == batch_idx).nonzero()[ : , 0]
                this_ct_y_true = torch.flatten(\
                        torch.index_select(batched_y_true, 0, this_ct_block_idx))
                this_ct_y_pred = torch.flatten(\
                        torch.index_select(batched_y_pred, 0, this_ct_block_idx))
                this_ct_pred_true = torch.stack(\
                        (this_ct_y_pred, this_ct_y_true), -1).cpu()
                ct_id = ct_id_list[batch_idx]
                io_worker_pool.apply_async(save_pred_true, \
                        args = (run_dirpath, ct_id, this_ct_pred_true, ))
    io_worker_pool.close()
    io_worker_pool.join()
    dist.destroy_process_group()


def get_current_time():
    time_now = datetime.datetime.now()
    time_now_timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
    return time_now, time_now_timestamp


if __name__ == '__main__':
    _, timestamp = get_current_time()
    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    run_dirpath = osp.join(snowcat_storage_dirpath, "inference", f"inference-{timestamp}")
    os.makedirs(run_dirpath, exist_ok=True)

    dataset_config = PICDataset.PICDatasetConfig(STI_DATA_DIRPATH, \
            SHORTCUT_EDGE_HOP, run_dirpath)

    test_ct_list = PICDataset.get_ct_list(CTI_LIST_FILEPATH, \
            STI_DATA_DIRPATH, DATASET_DIRPATH, SAMPLE_SIZE)
    print(f"loaded {len(test_ct_list)} CTs to predict")

    create_backup(run_dirpath, test_ct_list)

    test_dataset = PICDataset.PICDataset(test_ct_list, dataset_config)

    if USE_CPU is True:
        world_size = 4
        print('Let\'s use', world_size, type(world_size),'CPUs!')
    else:
        world_size = torch.cuda.device_count()
        print('Let\'s use', world_size, type(world_size),'GPUs!')
    args = (world_size, run_dirpath, test_dataset, )

    start_time, start_timestamp = get_current_time()
    mp.spawn(inference, args=args, nprocs=world_size, join=True)
    end_time, end_timestamp = get_current_time()

    time_diff = end_time - start_time
    with open(osp.join(run_dirpath, "time-diff"), "w") as time_diff_dump:
        print(time_diff, start_timestamp, end_timestamp, file=time_diff_dump)
