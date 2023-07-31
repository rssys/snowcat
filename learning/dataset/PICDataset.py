import os
import os.path as osp
import sys
import datetime
import torch
import traceback
import pickle
import random
from torch_geometric.data import Dataset, Data
from fairseq.data import Dictionary
# import custom modules
this_dirpath = os.path.abspath(os.path.dirname(__file__))
module_dirpath = osp.join(this_dirpath, "generator")
sys.path.append(module_dirpath)
import testinput
import graph
util_dirpath = osp.join(this_dirpath, "../util/")
import comm


class PICDatasetConfig():
    """
    A configuration of the PIC dataset:
    - sti_data_dirpath is the path of the sti exec data (e.g., control flow);
    - shortcut_edge_hop is the deneity of shortcut edges (e.g., every 8 hops);
    - truncate_tokens is the max number of tokens we keep for each code block;
    - run_dirpath is the path for this run of the PICDataset (e.g., store logs).
    """
    def __init__(self, sti_data_dirpath, shortcut_edge_hop=0, run_dirpath=None):
        self.sti_data_dirpath = sti_data_dirpath
        assert isinstance(shortcut_edge_hop, int) and shortcut_edge_hop >= 0
        self.shortcut_edge_hop = shortcut_edge_hop
        self.truncate_tokens = 24 # Limited by the GPU size
        assert os.path.exists(run_dirpath)
        self.run_dirpath = run_dirpath

    def report(self, num_ct):
        """ Print a log to the disk """
        # TODO: save a copy of CT list in this dataset
        report_filepath = osp.join(self.run_dirpath, "dataset-report")
        with open(report_filepath, "w") as f:
            time_now = datetime.datetime.now()
            timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
            f.write(f"report_time: {timestamp}\n")
            f.write(f"truncate_tokens: {self.truncate_tokens}\n")
            f.write(f"shortcut_edge_hop: {self.shortcut_edge_hop}\n")
            f.write(f"num_ct: {num_ct}\n")


PYG_GRAPH_SAVE_DIRPATH = osp.join(this_dirpath, "cached-pyg-cti-graph")
def find_cached_pyg_cti_graph(graph_name):
    graph_save_dirpath = osp.join(PYG_GRAPH_SAVE_DIRPATH, graph_name)
    pickle_filepath = osp.join(graph_save_dirpath, "graph.pickle")
    return comm.find_obj_cache(pickle_filepath)

def cache_pyg_cti_graph(graph_name, graph):
    graph_save_dirpath = osp.join(PYG_GRAPH_SAVE_DIRPATH, graph_name)
    os.makedirs(graph_save_dirpath, exist_ok=True)
    pickle_filepath = osp.join(graph_save_dirpath, "graph.pickle")
    comm.try_cache_obj(pickle_filepath, graph)


def add_schedule_edge(pyg_cti_graph, schedule_edge):
    edge_tensor = torch.tensor(schedule_edge, dtype=torch.int).t()
    pyg_ct_graph = pyg_cti_graph
    pyg_ct_graph.edge_index = \
            torch.cat((pyg_ct_graph.edge_index, edge_tensor), 1).contiguous()
    edge_type_tensor = torch.full((len(schedule_edge), 1), graph.SCHEDULE_EDGE)
    pyg_ct_graph.edge_attr = \
            torch.cat((pyg_ct_graph.edge_attr, edge_type_tensor), 0).contiguous()
    return pyg_ct_graph

def label_node(pyg_ct_graph, covered_node_idx_set):
    """
    num_pos_node_dict = {}
    for node_type in range(4):
        num_pos_node_dict[node_type] = 0
    """
    num_node = pyg_ct_graph.x.shape[0]
    y = torch.zeros([num_node, 1], dtype=torch.float)
    # TODO: If the performance can be improved
    for node_idx in covered_node_idx_set:
        y[node_idx] = 1
        """
        node_type = pyg_ct_graph.pos[node_idx].item()
        num_pos_node_dict[node_type] += 1
        """
    """
    print("label result", num_pos_node_dict)
    """
    labeled_pyg_ct_graph = pyg_ct_graph
    labeled_pyg_ct_graph.y = y
    return labeled_pyg_ct_graph


class PICDataset(Dataset):
    """
    The PIC dataset. It get() a pyg graph every time, which represents a CT.
    """
    def __init__(self, ct_list, config):
        for ct in ct_list:
            assert isinstance(ct, testinput.CT) is True
            break
        self.ct_list = ct_list
        assert isinstance(config, PICDatasetConfig)
        self.config = config
        block_token_dict = Dictionary()
        with open(osp.join(this_dirpath, "fairseq-dict.txt"), "r") as f: # code assembly token dict
            block_token_dict.add_from_file(f)
        self.block_token_dict = block_token_dict
        self.report()
        super().__init__(None, None, None, None)

    def report(self):
        """ Print a log to the disk (run_dirpath). """
        self.config.report(len(self.ct_list))

    def len(self):
        """ Number of CTs/graphs in this dataset. """
        return len(self.ct_list)

    def _convert_raw_graph(self, raw_graph):
        """ Convert a raw graph (type: generator/graph) to a pyg graph. """
        assert isinstance(raw_graph, graph.CTIGraph)
        num_node = len(raw_graph.node_list)
        tensor = torch.ones((2, ), dtype=torch.int)
        pad_encoding = self.block_token_dict.pad()
        node_token_idx_tensor = tensor.new_full((num_node, self.config.truncate_tokens), \
                pad_encoding, dtype=torch.int)
        node_type_tensor = torch.zeros((num_node, 1), dtype=torch.uint8)
        # TODO: improve the performance of the following block
        for node_idx, node in enumerate(raw_graph.node_list):
            # set its type in the tensor
            if node.cpu == "cpu0":
                if node.is_sequential is True:
                    node_type_tensor[node_idx] = 0
                else:
                    node_type_tensor[node_idx] = 2
            else:
                if node.is_sequential is True:
                    node_type_tensor[node_idx] = 1
                else:
                    node_type_tensor[node_idx] = 3
            # set its token idx
            node_assembly = node.block_assembly
            token_idx_list = self.block_token_dict.encode_line(\
                    "<s> " + node_assembly, add_if_not_exist=False)
            for pos, token_idx in enumerate(token_idx_list):
                if pos == self.config.truncate_tokens:
                    break
                node_token_idx_tensor[node_idx][pos] = token_idx
        raw_edge_list = []
        edge_type_list = []
        for edge in raw_graph.edge_list:
            raw_edge_list.append(edge.node_idx_pair)
            edge_type_list.append([edge.edge_type])

        edge_tuple_tensor = \
                torch.tensor(raw_edge_list, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_type_list, dtype=torch.int)
        #schedule_edge_idx_tensor = (edge_type_tensor == graph.SCHEDULE_EDGE).nonzero()[ : , 0]

        return Data(x=node_token_idx_tensor, \
                edge_index=edge_tuple_tensor, \
                edge_attr=edge_type_tensor, \
                pos=node_type_tensor, \
                y=None)

    def get(self, idx):
        try:
            ct = self.ct_list[idx]
            assert isinstance(ct, testinput.CT) is True
            # TODO: handle unexpected errors in graph.build_graph?
            raw_cti_graph = graph.build_cti_graph(ct.cti, self.config.sti_data_dirpath, \
                    self.config.shortcut_edge_hop)
            graph_name = raw_cti_graph.name
            pyg_cti_graph = find_cached_pyg_cti_graph(graph_name)
            if pyg_cti_graph is None:
                pyg_cti_graph = self._convert_raw_graph(raw_cti_graph)
                cache_pyg_cti_graph(graph_name, pyg_cti_graph)

            schedule_edge = graph.get_schedule_edge(ct, raw_cti_graph, self.config.sti_data_dirpath)
            pyg_ct_graph = add_schedule_edge(pyg_cti_graph, schedule_edge)

            covered_node_idx_set = raw_cti_graph.get_covered_node_idx(ct.coverage_by_cpu)
            labeled_pyg_ct_graph = label_node(pyg_ct_graph, covered_node_idx_set)

            return labeled_pyg_ct_graph, ct.id
        except:
            traceback.print_exc()
            error_dirpath = "./picdataset-error"
            os.makedirs(error_dirpath, exist_ok=True)
            log_filepath = osp.join(error_dirpath, f"worker-{os.getpid()}")
            with open(log_filepath, "a") as f:
                k = f"{ct.cti.sti_by_cpu['cpu0'].id} " \
                        f"{ct.cti.sti_by_cpu['cpu1'].id} " \
                        f"{ct.schedule.init_cpu} " \
                        f"{ct.schedule.cpu0_switch_point} " \
                        f"{ct.schedule.cpu1_switch_point}"
                print("error-encountered", k, file=f)
                traceback.print_exc(file=f)
            return Data(), -1


def worker_get_ct_list(cti_list, graph_info_dirpath, sample_size, ins_to_block_dict):
    ct_list = []
    for cti_id in cti_list:
        cti_data_dirpath = osp.join(graph_info_dirpath, cti_id, "schedule_info")
        if osp.exists(cti_data_dirpath) is False:
            continue
        cti_data_entry_list = list(os.scandir(cti_data_dirpath))
        random.shuffle(cti_data_entry_list)
        if sample_size is not None:
            if len(cti_data_entry_list) > sample_size:
                cti_data_entry_list = cti_data_entry_list[ : sample_size]
        unique_ct_id_set = set([])
        for f in cti_data_entry_list:
            info_split = f.name.split("_")
            sti_a = testinput.STI(info_split[0], "cpu0")
            sti_b = testinput.STI(info_split[1], "cpu1")
            cti = testinput.CTI(sti_a, sti_b)
            if info_split[2] == "0":
                init_cpu = "cpu0"
            else:
                init_cpu = "cpu1"
            cpu0_switch_point = int(info_split[3], 16)
            cpu1_switch_point = int(info_split[4], 16)
            try:
                cpu0_switch_point = ins_to_block_dict[cpu0_switch_point]
                cpu1_switch_point = ins_to_block_dict[cpu1_switch_point]
            except:
                continue
            #print(init_cpu, cpu0_switch_point, cpu1_switch_point)
            schedule = testinput.SCHEDULE(init_cpu, \
                    cpu0_switch_point, cpu1_switch_point)
            schedule_data_dirpath = osp.join(cti_data_dirpath, f.name)
            coverage_by_cpu = {}
            cpu0_coverage_filepath = \
                    osp.join(schedule_data_dirpath, "cpu0-coverage")
            cpu1_coverage_filepath = \
                    osp.join(schedule_data_dirpath, "cpu1-coverage")
            if osp.exists(cpu0_coverage_filepath) is False or \
                    osp.exists(cpu1_coverage_filepath) is False:
                        continue
            coverage_by_cpu["cpu0"] = cpu0_coverage_filepath
            coverage_by_cpu["cpu1"] = cpu1_coverage_filepath
            ct = testinput.CT(cti, schedule, coverage_by_cpu)
            if ct.id in unique_ct_id_set:
                continue
            unique_ct_id_set.add(ct.id)
            ct_list.append(ct)
    return ct_list


def get_ct_list(cti_list_filepath, sti_data_dirpath, graph_info_dirpath, sample_size):
    cti_set = set([])
    with open(cti_list_filepath, "r") as tmp_file:
        for line in tmp_file:
            cti_id = line.strip()
            cti_set.add(cti_id)
    cti_list = list(cti_set)
    block_range_list = []
    ins_to_block_dict = {}
    kernel_info_dirpath = os.environ.get('KERNEL_INFO_DIR')
    if kernel_info_dirpath is None:
        print("Please source choose_kernel.sh")
        exit(1)

    vmlinux_map_filepath = osp.join(kernel_info_dirpath, "vmlinux.map")
    with open(vmlinux_map_filepath, "r") as vmlinux_map_file:
        for line in vmlinux_map_file:
            ins_addr = int(line.strip()[ : 10], 16)
            ins_to_block_dict[ins_addr] = -1

    block_info_filepath = osp.join(kernel_info_dirpath, "block-info")
    with open(block_info_filepath, "r") as block_info_file:
        for line in block_info_file:
            line_split = line.strip().split(" ")
            block_start_addr = int(line_split[0], 16)
            block_end_addr = int(line_split[1], 16)
            block_range = tuple([block_start_addr, block_end_addr])
            block_range_list.append(block_range)
            for ins_addr in range(block_start_addr, block_end_addr):
                if ins_addr not in ins_to_block_dict:
                    continue
                ins_to_block_dict[ins_addr] = block_start_addr
    del block_range_list

    ct_list = worker_get_ct_list(cti_list, graph_info_dirpath, \
            sample_size, ins_to_block_dict)
    return ct_list
