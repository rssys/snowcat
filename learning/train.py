import os
import os.path as osp
import sys
import datetime
import configparser
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torchmetrics import MeanMetric
#import custom modules
import model.model as Model
from dataset.sampler import DistributedEvalSampler
from dataset import PICDataset
from evaluation.metric import GraphAveragePrecision


config_filepath = sys.argv[1]
if osp.exists(config_filepath) is not True:
    print(f"config file {config_filepath} does not exist")
    exit(-1)

# Configure the trainning based on the user config
config = configparser.ConfigParser()
config.read(config_filepath)
DATASET_DIRPATH = config["dataset"]["dataset_dirpath"]
DATASET_SPLIT_DIRPATH = config["dataset"]["dataset_split_dirpath"]
SHORTCUT_EDGE_HOP = int(config["dataset"]["shortcut_edge_hop"])
if config["dataset"]["num_ct_per_cti"].lower() == "none":
    SAMPLE_SIZE = None
else:
    SAMPLE_SIZE = int(config["dataset"]["num_ct_per_cti"])
STI_DATA_DIRPATH = config["dataset"]["sti_data_dirpath"]

PRETRAINED_BERT_FILEPATH = config["model"]["pretrained_bert_ckpt_filepath"]
assert osp.exists(PRETRAINED_BERT_FILEPATH) is True
CHECKPOINT_FILEPATH = config["model"]["checkpoint_filepath"]
if CHECKPOINT_FILEPATH == "None":
    CHECKPOINT_FILEPATH = None

BATCH_SIZE = int(config["train"]["batch_size"])
UPDATE_FREQ = int(config["train"]["update_freq"])
LEARNING_RATE = float(config["train"]["learning_rate"])
NUM_EPOCH = int(config["train"]["num_epoch"])
LOSS_FN = config["train"]["loss_fn"]
USE_CPU = config["train"].getboolean("use_cpu")
USE_AMP = config["train"].getboolean("use_amp")
if USE_CPU is True and USE_AMP is True:
    print("AMP cannot be used for CPU training")
    exit(0)
if USE_AMP is True:
    import apex
    from apex import amp
FINE_TUNING = config["train"].getboolean("fine_tuning")


def save_arch_info(run_dirpath, model):
    """Save some model info."""
    model_arch_filepath = osp.join(run_dirpath, "model-arch")
    with open(model_arch_filepath, "w") as model_arch_file:
        print(model, file=model_arch_file)
    parameter_filepath = osp.join(run_dirpath, "bert-parameters")
    with open(parameter_filepath, "w") as parameter_file:
        for param in model.bert.model.parameters():
            print(param.name, param.data, file=parameter_file)


def create_backup(run_dirpath, train_ct_list, val_ct_list):
    """Create a backup for reproducing results."""
    backup_dirpath = osp.join(run_dirpath, "backup")
    os.makedirs(backup_dirpath, exist_ok=True)
    cmd = f"cp {config_filepath} {backup_dirpath}/"
    os.system(cmd)
    cmd = f"cp train.py {backup_dirpath}/"
    os.system(cmd)
    cmd = f"cp dataset/PICDataset.py {backup_dirpath}"
    os.system(cmd)
    with open(osp.join(backup_dirpath, "train_ct_list"), "w") as tmp_file:
        for ct in train_ct_list:
            print(ct.id, file=tmp_file)
    with open(osp.join(backup_dirpath, "val_ct_list"), "w") as tmp_file:
        for ct in val_ct_list:
            print(ct.id, file=tmp_file)


def save_checkpoint(run_dirpath, epoch, model, optimizer, scheduler):
    """Save a checkpoint of the model."""
    if USE_AMP is True:
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'amp': amp.state_dict()
            }
        checkpoint_filepath = osp.join(run_dirpath, f"amp-checkpoint-{epoch}.tar")
    else:
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
        checkpoint_filepath = osp.join(run_dirpath, f"amp-checkpoint-{epoch}.tar")
    torch.save(checkpoint, checkpoint_filepath)
    print(f"Saved checkpoint to {checkpoint_filepath}")


def train(rank, world_size, run_dirpath, train_dataset, val_dataset):
    """Trainer code for each GPU."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist_protocol = "gloo" if USE_CPU is True else "nccl"
    dist.init_process_group(dist_protocol, rank=rank, world_size=world_size)

    # One can increase/decrease the value of `num_workers` to 
    # control the cpu pressure. We used the value 48 on our machines.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, \
            rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, \
            sampler=train_sampler, pin_memory=True, num_workers=12, prefetch_factor=64)
    num_train_graph = len(train_dataset)
    num_train_graph_per_gpu = int(num_train_graph / world_size)

    val_sampler = DistributedEvalSampler(val_dataset, num_replicas=world_size, \
            rank=rank, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, \
            sampler=val_sampler, pin_memory=True, num_workers=12, prefetch_factor=64)
    num_val_graph = len(val_dataset)
    num_val_graph_per_gpu = int(num_val_graph / world_size)

    if USE_CPU is False:
        device = torch.device('cuda:%d' % rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    torch.manual_seed(12345)
    start_epoch = 1

    '''Create the model either from scratch or a checkpoint.'''
    model = Model.create_model(config["model"])
    model.to(device)
    if rank == 0:
        save_arch_info(run_dirpath, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    if USE_AMP is True:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if CHECKPOINT_FILEPATH is not None:
        if USE_CPU is False:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            model_checkpoint = torch.load(CHECKPOINT_FILEPATH, map_location=map_location)
        else:
            model_checkpoint = torch.load(CHECKPOINT_FILEPATH, map_location=torch.device('cpu'))
        model.load_state_dict(model_checkpoint["model"])
        if USE_AMP is True:
            amp.load_state_dict(model_checkpoint["amp"])
        if FINE_TUNING is False:
            optimizer.load_state_dict(model_checkpoint["optimizer"])
            scheduler.load_state_dict(model_checkpoint["scheduler"])
            start_epoch = model_checkpoint["epoch"] + 1
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
            if USE_AMP is True:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    """Distribute the model."""
    if USE_AMP is True:
        assert USE_CPU is False
        model = torch.nn.parallel.DistributedDataParallel(model, \
                device_ids=[rank], output_device=rank, find_unused_parameters=True)
    else:
        if USE_CPU is False:
            model = torch.nn.parallel.DistributedDataParallel(model, \
                    device_ids=[rank], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, \
                    device_ids=None, find_unused_parameters=True)


    # Save the initial random model as the baseline
    if rank == 0 and start_epoch == 1:
        save_checkpoint(run_dirpath, 0, model, optimizer, scheduler)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loss = MeanMetric().to(device)
    train_ap_on_all = GraphAveragePrecision().to(device)
    train_ap_on_ur = GraphAveragePrecision().to(device)
    val_loss = MeanMetric().to(device)
    val_ap_on_all = GraphAveragePrecision().to(device)
    val_ap_on_ur = GraphAveragePrecision().to(device)

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        """Start training"""
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        train_ap_on_all.reset()
        train_ap_on_ur.reset()
        train_loss.reset()

        num_trained_graph = 0

        train_start_time = datetime.datetime.now()
        for data, graph_index in train_loader:
            # the trainning dataset should not contain any annoying CT graphs
            # TODO: add more comments
            assert graph_index != -1

            block_token = data.x.to(device)
            edge  = data.edge_index.to(device)
            edge_type = data.edge_attr.to(device)
            y_true = data.y.to(device)
            block_type = data.pos.to(device)

            y_pred = model(block_token, edge, edge_type)

            loss = loss_fn(y_pred, y_true) / (UPDATE_FREQ * world_size)
            if USE_AMP is True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            num_trained_graph += data.num_graphs
            if num_trained_graph % UPDATE_FREQ == 0:
                optimizer.step()
                optimizer.zero_grad()

            y_pred = torch.sigmoid(y_pred)
            # 1. Average precision on all blocks
            y_true = y_true.int()
            train_ap_on_all(y_pred, y_true)

            # 2. Average precision on UR blocks
            ur_block_index = (block_type >= 2).nonzero()[ : , 0]
            ur_y_pred = torch.index_select(y_pred, 0, ur_block_index)
            ur_y_true = torch.index_select(y_true, 0, ur_block_index)
            train_ap_on_ur(ur_y_pred, ur_y_true)

            loss_value = loss.item()
            train_loss.update(loss_value * data.num_graphs)

            if num_trained_graph % 100 == 0:
                ct = datetime.datetime.now()
                avg_ap_on_all = train_ap_on_all.compute().item()
                avg_ap_on_ur = train_ap_on_ur.compute().item()
                avg_loss = train_loss.compute().item()
                last_lr_rate = scheduler.get_last_lr()
                print(ct, f"epoch: {epoch:02d} loss: {avg_loss} rank: {rank:02d} " \
                        f"update_frequency: {UPDATE_FREQ} " \
                        f"trained_graphs_this_gpu: {num_trained_graph} " \
                        f"total_graphs_per_gpu: {num_train_graph_per_gpu} " \
                        f"train_ap_on_all: {avg_ap_on_all} " \
                        f"train_ap_on_ur: {avg_ap_on_ur} " \
                        f"last_lr_rate: {last_lr_rate}")

        train_loss_avg = train_loss.compute().item()
        train_ap_on_all_avg = train_ap_on_all.compute().item()
        train_ap_on_ur_avg = train_ap_on_ur.compute().item()

        dist.barrier()
        if rank == 0:
            save_checkpoint(run_dirpath, epoch, model, optimizer, scheduler)
        else:
            print(f"Rank-{rank} waiting rank-0 for saving the checkpoint")

        train_end_time= datetime.datetime.now()
        dist.barrier()

        """Start the validation."""
        val_sampler.set_epoch(epoch)
        model.eval()

        num_validated_graph = 0
        val_ap_on_all.reset()
        val_ap_on_ur.reset()
        val_loss.reset()

        for data, graph_index in val_loader:
            if graph_index == -1:
                continue
            with torch.no_grad():
                block_token = data.x.to(device)
                edge = data.edge_index.to(device)
                edge_type = data.edge_attr.to(device)
                y_true = data.y.to(device)
                block_type = data.pos.to(device)
                y_pred = model.module(block_token, edge, edge_type)
                loss = loss_fn(y_pred, y_true)
                val_loss.update(loss.item())

                y_pred = torch.sigmoid(y_pred)
                y_true = y_true.to(torch.int)

                # 1. Average precision on all blocks
                val_ap_on_all(y_pred, y_true)
                # 2. Average precision on UR blocks
                ur_block_index = (block_type >= 2).nonzero()[ : , 0]
                ur_y_pred = torch.index_select(y_pred, 0, ur_block_index)
                ur_y_true = torch.index_select(y_true, 0, ur_block_index)
                val_ap_on_ur(ur_y_pred, ur_y_true)

                num_validated_graph += data.num_graphs

                if num_validated_graph % 100 == 0:
                    ct = datetime.datetime.now()
                    avg_ap_on_all = val_ap_on_all.compute().item()
                    avg_ap_on_ur = val_ap_on_ur.compute().item()
                    avg_loss = val_loss.compute().item()
                    last_lr_rate = scheduler.get_last_lr()
                    print(ct, f"epoch: {epoch:02d} loss: {avg_loss} rank: {rank:02d} " \
                            f"validated_graphs_this_gpu: {num_validated_graph} " \
                            f"total_graphs_per_gpu: {num_val_graph_per_gpu} " \
                            f"val_ap_on_all: {avg_ap_on_all} " \
                            f"val_ap_on_ur: {avg_ap_on_ur} " \
                            f"last_lr_rate: {last_lr_rate}")

        val_loss_avg = val_loss.compute().item()
        val_ap_on_all_avg = val_ap_on_all.compute().item()
        val_ap_on_ur_avg = val_ap_on_ur.compute().item()
        if rank == 0:
            validation_log = open(osp.join(run_dirpath, "validation"), "a")
            last_lr_rate = scheduler.get_last_lr()
            epoch_train_time = train_end_time - train_start_time
            print(f"epoch: {epoch} last_lr_rate: {last_lr_rate} " \
                    f"train_loss: {train_loss_avg} " \
                    f"train_ap_on_all: {train_ap_on_all_avg} " \
                    f"train_ap_on_ur: {train_ap_on_ur_avg} ", \
                    f"validation_loss: {val_loss_avg} " \
                    f"val_ap_on_all: {val_ap_on_all_avg} " \
                    f"val_ap_on_ur: {val_ap_on_ur_avg} ", \
                    f"epoch_train_time: {epoch_train_time}", \
                    file=validation_log)
            validation_log.close()

        scheduler.step()
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    run_dirpath = osp.join(snowcat_storage_dirpath, "training", f"train-{timestamp}")
    os.makedirs(run_dirpath, exist_ok=True)

    dataset_config = PICDataset.PICDatasetConfig(STI_DATA_DIRPATH, \
            SHORTCUT_EDGE_HOP, run_dirpath)

    train_cti_list_filepath = osp.join(DATASET_SPLIT_DIRPATH, "train_cti_list")
    train_ct_list = PICDataset.get_ct_list(train_cti_list_filepath, \
            STI_DATA_DIRPATH, DATASET_DIRPATH, SAMPLE_SIZE)
    print(f"loaded {len(train_ct_list)} CTs for train")

    val_cti_list_filepath = osp.join(DATASET_SPLIT_DIRPATH, "val_cti_list")
    val_ct_list = PICDataset.get_ct_list(val_cti_list_filepath, \
            STI_DATA_DIRPATH, DATASET_DIRPATH, SAMPLE_SIZE)
    print(f"loaded {len(val_ct_list)} CTs for val")

    create_backup(run_dirpath, train_ct_list, val_ct_list)

    train_dataset = PICDataset.PICDataset(train_ct_list, dataset_config)
    val_dataset = PICDataset.PICDataset(val_ct_list, dataset_config)

    if USE_CPU is False:
        world_size = torch.cuda.device_count()
        print('Let\'s use', world_size, type(world_size),'GPUs!')
    else:
        world_size = 2
        print('Let\'s use', world_size, type(world_size),'CPUs!')
    args = (world_size, run_dirpath, train_dataset, val_dataset, )
    mp.spawn(train, args=args, nprocs=world_size, join=True)
