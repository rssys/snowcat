import os
import os.path as osp
import pickle
import sys
# import custom modules
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
sys.path.append(osp.join(main_home, "script", "data-collection", \
        "sti-data"))
import datafetcher
sys.path.append(osp.join(main_home, "learning", "dataset", "generator"))
import testinput
sys.path.append(osp.join(main_home, "learning", "util"))
import comm


class Node():
    def __init__(self, block_addr, cpu, is_sequential, hop=None):
        self.block_addr = block_addr
        self.block_assembly = None
        assert cpu in ["cpu0", "cpu1"]
        self.cpu = cpu
        self.is_sequential = is_sequential
        if hop != None:
            assert self.is_sequential is False
        self.hop = hop

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        if self.block_addr != getattr(cmp, "block_addr"):
            return False
        if self.cpu != getattr(cmp, "cpu"):
            return False
        return True

    def __hash__(self):
        # The attribute is_sequential is not used in the hash
        # because we only want to keep unique code blocks in the graph.
        # For example, doing this would ensure all URBs are unseen
        return hash(str(self.block_addr) + self.cpu)


""" Different types of edges """
SEQUENTIAL_CONTROL_EDGE = 0
POSSIBLE_CONTROL_EDGE = 1
INTRA_DATA_FLOW_EDGE = 2
INTER_DATA_FLOW_EDGE = 3
SCHEDULE_EDGE = 4
SHORTCUT_EDGE = 5

class Edge():
    def __init__(self, node_idx_pair, edge_type):
        assert len(node_idx_pair) == 2
        self.node_idx_pair = node_idx_pair
        assert SEQUENTIAL_CONTROL_EDGE <= edge_type <= SHORTCUT_EDGE
        self.edge_type = edge_type


class CTIGraph():
    def __init__(self, cti, shortcut_edge_hop):
        self.node_list = []
        self.edge_list = []
        self.node_lookup_dict = {}
        self.name = ""
        self.cti = cti
        self.init_graph_name(cti, shortcut_edge_hop)

    def init_graph_name(self, cti, shortcut_edge_hop):
        sti_cpu0 = cti.sti_by_cpu["cpu0"]
        sti_cpu1 = cti.sti_by_cpu["cpu1"]
        self.name = f"{sti_cpu0.id}_{sti_cpu1.id}.shortcut_hop-{shortcut_edge_hop}"

    def __getstate__(self):
        return (self.node_list, self.edge_list, self.node_lookup_dict, \
                self.name, self.cti)

    def __setstate__(self, state):
        self.node_list, self.edge_list, self.node_lookup_dict, \
                self.name, self.cti = state

    def try_get_node_idx(self, block_addr, cpu, is_sequential, hop):
        """
        Return the idx of the node in self.node_list if the target node exists,
        return none if the target node does not exist.
        """
        node = Node(block_addr, cpu, is_sequential, hop)
        if node not in self.node_lookup_dict:
            return None
        return self.node_lookup_dict[node]

    def get_node_idx(self, block_addr, cpu, is_sequential, hop):
        """
        Return the idx of the node in self.node_list.
        If the target node does not exist, create one.
        """
        node = Node(block_addr, cpu, is_sequential, hop)
        if node not in self.node_lookup_dict:
            self.node_list.append(node)
            self.node_lookup_dict[node] = len(self.node_list) - 1
        return self.node_lookup_dict[node]

    def _add_edge(self, edge_set, edge_type):
        """ Add a list of edges to self.edge_list. """
        for node_idx_pair in edge_set:
            edge = Edge(node_idx_pair, edge_type)
            self.edge_list.append(edge)

    def add_sc_control_flow(self, sc_control_flow, cpu):
        """ Add the sc control flow as edges to the graph. """
        sc_control_edge_set = set([])
        for flow in sc_control_flow:
            src = flow.src
            dst = flow.dst
            assert src.cpu == dst.cpu == cpu
            # a code block turns into a node now
            src_node_idx = self.get_node_idx(src.block_addr, cpu, \
                    is_sequential=True, hop=None)
            dst_node_idx = self.get_node_idx(dst.block_addr, cpu, \
                    is_sequential=True, hop=None)
            edge = tuple([src_node_idx, dst_node_idx])
            sc_control_edge_set.add(edge)
        self._add_edge(sc_control_edge_set, SEQUENTIAL_CONTROL_EDGE)

    def add_ur_control_flow(self, ur_control_flow, cpu):
        """ Add the ur control flow as edges to the graph. """
        ur_control_edge_set = set([])
        for hop in ur_control_flow:
            ur_control_flow_this_hop = ur_control_flow[hop]
            for flow in ur_control_flow_this_hop:
                src = flow.src
                dst = flow.dst
                assert src.cpu == dst.cpu == cpu
                # The src node (code block) must already exist before
                # adding the UR blocks. If hop==1, the src node should be
                # a SC node and if hop > 1, they should be the UR blocks
                # from the last hop.
                if hop == 1:
                    src_node_idx = self.try_get_node_idx(src.block_addr, cpu, \
                            is_sequential=True, hop=None)
                else:
                    src_node_idx = self.try_get_node_idx(src.block_addr, cpu, \
                            is_sequential=False, hop=hop - 1)
                assert src_node_idx is not None
                dst_node_idx = self.get_node_idx(dst.block_addr, cpu, \
                        is_sequential=False, hop=hop)
                edge = tuple([src_node_idx, dst_node_idx])
                ur_control_edge_set.add(edge)
        self._add_edge(ur_control_edge_set, POSSIBLE_CONTROL_EDGE)

    def add_intra_data_flow(self, intra_data_flow, cpu):
        """ Add the intra-thread data flow as edges to the graph. """
        intra_data_edge_set = set([])
        for data_flow in intra_data_flow:
            src = data_flow.src
            dst = data_flow.dst
            assert src.cpu == dst.cpu == cpu
            src_node_idx = self.try_get_node_idx(src.block_addr, cpu, \
                    is_sequential=True, hop=None)
            dst_node_idx = self.try_get_node_idx(dst.block_addr, cpu, \
                    is_sequential=True, hop=None)
            if src_node_idx is None or dst_node_idx is None:
                continue
            # the data flow should only happen between SC code blocks / nodes
            src_node = self.node_list[src_node_idx]
            dst_node = self.node_list[dst_node_idx]
            if src_node.is_sequential is False or dst_node.is_sequential is False:
                continue
            edge = tuple([src_node_idx, dst_node_idx])
            intra_data_edge_set.add(edge)
        self._add_edge(intra_data_edge_set, INTRA_DATA_FLOW_EDGE)

    def add_inter_data_flow(self, inter_data_flow):
        """ Add the inter-thread data flow as edges to the graph. """
        inter_data_edge_set = set([])
        for data_flow in inter_data_flow:
            src = data_flow.src
            dst = data_flow.dst
            assert src.cpu != dst.cpu
            src_node_idx = self.try_get_node_idx(src.block_addr, src.cpu, \
                    is_sequential=True, hop=None)
            dst_node_idx = self.try_get_node_idx(dst.block_addr, dst.cpu, \
                    is_sequential=True, hop=None)
            if src_node_idx is None or dst_node_idx is None:
                continue
            # the data flow should only happen between SC code blocks / nodes
            src_node = self.node_list[src_node_idx]
            dst_node = self.node_list[dst_node_idx]
            if src_node.is_sequential is False or dst_node.is_sequential is False:
                continue
            edge = tuple([src_node_idx, dst_node_idx])
            inter_data_edge_set.add(edge)
        self._add_edge(inter_data_edge_set, INTER_DATA_FLOW_EDGE)

    def add_shortcut_edge(self, shortcut_flow, cpu):
        """ Add the shortcut edges to encourage message passing. """
        shortcut_edge_set = set([])
        for flow in shortcut_flow:
            src = flow.src
            dst = flow.dst
            assert src.cpu == dst.cpu == cpu
            # a code block turns into a node now
            src_node_idx = self.try_get_node_idx(src.block_addr, cpu, \
                    is_sequential=True, hop=None)
            dst_node_idx = self.try_get_node_idx(dst.block_addr, cpu, \
                    is_sequential=True, hop=None)
            if src_node_idx is None or dst_node_idx is None:
                continue
            edge = tuple([src_node_idx, dst_node_idx])
            shortcut_edge_set.add(edge)
        self._add_edge(shortcut_edge_set, SHORTCUT_EDGE)

    def init_block_assembly(self, block_assembly_dict):
        for node in self.node_list:
            if node.block_assembly is not None:
                assert isinstance(node.block_assembly, str)
                continue
            node.block_assembly = block_assembly_dict[node.block_addr]

    def get_schedule_edge(self, schedule_flow):
        """ Add the scheduling hint as edges to the graph. """
        # TODO: currently assume the schedule is set at the block level\
        # ---it consists of two block addresses. Need to think how to take ins-level schedule.
        schedule_edge_set = set([])
        for flow in schedule_flow:
            src = flow.src
            dst = flow.dst
            assert src.cpu != dst.cpu
            src_node_idx = self.try_get_node_idx(src.block_addr, src.cpu, \
                    is_sequential=True, hop=None)
            dst_node_idx = self.try_get_node_idx(dst.block_addr, dst.cpu, \
                    is_sequential=True, hop=None)
            if src_node_idx is None or dst_node_idx is None:
                continue
            edge = tuple([src_node_idx, dst_node_idx])
            schedule_edge_set.add(edge)
        return list(schedule_edge_set)

    def get_covered_node_idx(self, block_coverage_by_cpu):
        assert block_coverage_by_cpu is not None
        covered_node_idx_set = set([])
        for cpu in ["cpu0", "cpu1"]:
            with open(block_coverage_by_cpu[cpu], "rb") as f:
                block_coverage = pickle.load(f)
            for block_addr in block_coverage:
                node_idx = self.try_get_node_idx(block_addr, cpu, \
                        is_sequential=True, hop=None)
                if node_idx is None:
                    continue
                covered_node_idx_set.add(node_idx)
        return covered_node_idx_set

    def save_node_info(self, save_filepath):
        num_node_dict = {"cpu0_sc" : 0, "cpu0_ur" : 0, "cpu1_sc" : 0, "cpu1_ur" : 0}
        for node in self.node_list:
            if node.cpu == "cpu0":
                if node.is_sequential is True:
                    num_node_dict["cpu0_sc"] += 1
                else:
                    num_node_dict["cpu0_ur"] += 1
            else:
                if node.is_sequential is True:
                    num_node_dict["cpu1_sc"] += 1
                else:
                    num_node_dict["cpu1_ur"] += 1
        msg_split = []
        for node_type in num_node_dict:
            msg_split.append(f"{node_type}: {num_node_dict[node_type]}")
        with open(save_filepath, "w") as f:
            print(" ".join(msg_split), file=f)

    def save_edge_info(self, save_filepath):
        num_edge_dict = {}
        for edge in self.edge_list:
            edge_type_str = None
            if edge.edge_type == SEQUENTIAL_CONTROL_EDGE:
                edge_type_str = "SEQUENTIAL_CONTROL_EDGE"
            elif edge.edge_type == POSSIBLE_CONTROL_EDGE:
                edge_type_str = "POSSIBLE_CONTROL_EDGE"
            elif edge.edge_type == INTRA_DATA_FLOW_EDGE:
                edge_type_str = "INTRA_DATA_FLOW_EDGE"
            elif edge.edge_type == INTER_DATA_FLOW_EDGE:
                edge_type_str = "INTER_DATA_FLOW_EDGE"
            elif edge.edge_type == SCHEDULE_EDGE:
                edge_type_str = "SCHEDULE_EDGE"
            elif edge.edge_type == SHORTCUT_EDGE:
                edge_type_str = "SHORTCUT_EDGE"
            else:
                assert 0
            if edge_type_str not in num_edge_dict:
                num_edge_dict[edge_type_str] = 0
            num_edge_dict[edge_type_str] += 1
        msg_split = []
        for edge_type in num_edge_dict:
            msg_split.append(f"{edge_type}: {num_edge_dict[edge_type]}")
        with open(save_filepath, "w") as f:
            print(" ".join(msg_split), file=f)


this_dirpath = osp.abspath(osp.dirname(__file__))
GRAPH_SAVE_DIRPATH = osp.join(this_dirpath, "cached-raw-cti-graph")
def find_cached_graph(graph_name):
    graph_save_dirpath = osp.join(GRAPH_SAVE_DIRPATH, graph_name)
    pickle_filepath = osp.join(graph_save_dirpath, "graph.pickle")
    return comm.find_obj_cache(pickle_filepath)

def cache_graph(graph_name, graph, debug=False):
    graph_save_dirpath = osp.join(GRAPH_SAVE_DIRPATH, graph_name)
    os.makedirs(graph_save_dirpath, exist_ok=True)
    pickle_filepath = osp.join(graph_save_dirpath, "graph.pickle")
    comm.try_cache_obj(pickle_filepath, graph)
    if debug is False:
        return
    node_info_filepath = osp.join(graph_save_dirpath, "node_info")
    graph.save_node_info(node_info_filepath)
    edge_info_filepath = osp.join(graph_save_dirpath, "edge_info")
    graph.save_edge_info(edge_info_filepath)


def build_cti_graph(cti, sti_data_dirpath, shortcut_edge_hop=0):
    assert isinstance(cti, testinput.CTI)
    assert isinstance(shortcut_edge_hop, int) and shortcut_edge_hop >= 0

    graph = CTIGraph(cti, shortcut_edge_hop)
    graph_name = graph.name
    cached_graph = find_cached_graph(graph_name)
    if cached_graph is None:
        kernel_helper = datafetcher.DataFetcher(sti_data_dirpath)
        for cpu in ["cpu0", "cpu1"]:
            sti = cti.sti_by_cpu[cpu]
            # Add nodes and edges for control flows
            sc_control_flow = kernel_helper.get_sc_control_flow(sti)
            graph.add_sc_control_flow(sc_control_flow, cpu)

            # Add shortcut edges to encourage message passing
            if shortcut_edge_hop != 0:
                shortcut_flow = kernel_helper.get_shortcut_flow(sti, shortcut_edge_hop)
                graph.add_shortcut_edge(shortcut_flow, cpu)

            ur_control_flow = kernel_helper.get_ur_control_flow(sti)
            graph.add_ur_control_flow(ur_control_flow, cpu)

            # Add edges for data flows
            sti_intra_data_flow = kernel_helper.get_intra_data_flow(sti)
            graph.add_intra_data_flow(sti_intra_data_flow, cpu)

        # Find the code assembly for each code block
        cpu0_sti = cti.sti_by_cpu["cpu0"]
        cpu0_block_assembly_dict = kernel_helper.get_block_assembly_dict(cpu0_sti)
        cpu1_sti = cti.sti_by_cpu["cpu1"]
        cpu1_block_assembly_dict = kernel_helper.get_block_assembly_dict(cpu1_sti)
        block_assembly_dict = cpu0_block_assembly_dict | cpu1_block_assembly_dict
        graph.init_block_assembly(block_assembly_dict)

        # Add the inter-thread data flows between two CPUs/threads
        inter_data_flow = kernel_helper.get_inter_data_flow(cti)
        graph.add_inter_data_flow(inter_data_flow)

        cache_graph(graph.name, graph, debug=True)
    else:
        graph = cached_graph
        assert graph.cti == cti
    return graph

def get_schedule_edge(ct, cti_graph, sti_data_dirpath):
    kernel_helper = datafetcher.DataFetcher(sti_data_dirpath)
    schedule_flow = kernel_helper.get_schedule_flow(ct)
    return cti_graph.get_schedule_edge(schedule_flow)
