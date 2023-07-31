import os
import os.path as osp
import pickle
import logging
import re
import sys
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
sys.path.append(osp.join(main_home, "script", "data-collection", "sti-data"))
import databuilder
sys.path.append(osp.join(main_home, "learning", "dataset", "generator"))
import testinput


class FlowEndpoint():
    """ Two endpoints make up a flow. """
    def __init__(self, block_addr, running_cpu):
        assert running_cpu in ["cpu0", "cpu1"]
        assert isinstance(block_addr, int)
        self.block_addr = block_addr
        self.cpu = running_cpu


class Flow():
    """ A flow connects two endpoints (block_addr + cpu) """
    def __init__(self, src_block_addr, dst_block_addr, src_cpu, dst_cpu):
        self.src = FlowEndpoint(src_block_addr, src_cpu)
        self.dst = FlowEndpoint(dst_block_addr, dst_cpu)

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        cmp_src = getattr(cmp, "src", None)
        cmp_dst = getattr(cmp, "dst", None)
        if cmp_src is None or cmp_dst is None:
            return False
        if cmp_src.block_addr != self.src.block_addr or \
                cmp_dst.block_addr != self.dst.block_addr or \
                cmp_src.cpu != self.src.cpu or cmp_dst.cpu != self.dst.cpu:
            return False
        return True

    def __hash__(self):
        return hash(str(self.src.block_addr) + self.src.cpu + \
                str(self.dst.block_addr) + self.dst.cpu)

    def show(self):
        """ Display the flow """
        print(f"src_block: {hex(self.src.block_addr)} src_cpu: {self.src.cpu} ---> " \
                f"dst_block: {hex(self.dst.block_addr)} dst_cpu: {self.dst.cpu}")


class DataFetcher():
    """ Fetch data extracted from executions of STIs """

    def __init__(self, sti_data_dirpath):
        self.sti_data_dirpath = sti_data_dirpath
        self.check_fetcher_prerequisite()

    def check_fetcher_prerequisite(self):
        """ The data must exist for the fetcher to work """
        necessary_dirname_list = ["sc_control_flow", "ur_control_flow", \
                "code_block_sequence", "block_assembly_dict", \
                "intra_data_flow", "shared_mem_access"]
        for necessary_dirname in necessary_dirname_list:
            necessary_dirpath = osp.join(self.sti_data_dirpath, necessary_dirname)
            assert osp.exists(necessary_dirpath) is True

    def get_sc_control_flow(self, sti):
        """ Load the pre-generated sc control flow """
        assert isinstance(sti, testinput.STI)
        pickle_filepath = osp.join(self.sti_data_dirpath, \
                "sc_control_flow", sti.id, sti.cpu)
        if osp.exists(pickle_filepath) is False:
            return None
        with open(pickle_filepath, "rb") as pickle_file:
            sc_control_flow = pickle.load(pickle_file)
        return sc_control_flow

    def get_ur_control_flow(self, sti):
        """ Load the pre-generated ur control flow """
        assert isinstance(sti, testinput.STI)
        pickle_filepath = osp.join(self.sti_data_dirpath, \
                "ur_control_flow", sti.id, sti.cpu)
        if osp.exists(pickle_filepath) is False:
            return None
        with open(pickle_filepath, "rb") as pickle_file:
            ur_control_flow = pickle.load(pickle_file)
        return ur_control_flow

    def get_block_assembly_dict(self, sti):
        """ Load the dict that returns the assembly given a block addr """
        assert isinstance(sti, testinput.STI)
        pickle_filepath = osp.join(self.sti_data_dirpath, \
                "block_assembly_dict", sti.id, sti.cpu)
        with open(pickle_filepath, "rb") as pickle_file:
            block_assembly_dict = pickle.load(pickle_file)
        return block_assembly_dict

    def get_intra_data_flow(self, sti):
        """ Load the pre-generated intra data flow """
        assert isinstance(sti, testinput.STI)
        pickle_filepath = osp.join(self.sti_data_dirpath, "intra_data_flow", \
                sti.id, sti.cpu)
        pickle_file = open(pickle_filepath, "rb")
        with open(pickle_filepath, "rb") as pickle_file:
            intra_data_flow = pickle.load(pickle_file)
        return intra_data_flow

    def get_inter_data_flow(self, cti):
        """ Predict the inter thread data flow between STIs
        (The approach is motivated by Snowboard) """
        assert isinstance(cti, testinput.CTI)

        inter_data_flow = set([])
        shared_mem_access_dict_by_cpu = {}
        for cpu in ["cpu0", "cpu1"]:
            pickle_filepath = osp.join(self.sti_data_dirpath, "shared_mem_access", \
                    cti.sti_by_cpu[cpu].id, cti.sti_by_cpu[cpu].cpu)
            with open(pickle_filepath, "rb") as pickle_file:
                shared_mem_access_dict_by_cpu[cpu] = pickle.load(pickle_file)

        common_mem_addr_set = set(shared_mem_access_dict_by_cpu["cpu0"].keys()) & \
                set(shared_mem_access_dict_by_cpu["cpu1"].keys())
        for mem_addr in common_mem_addr_set:
            cpu0_write_block_set = \
                    shared_mem_access_dict_by_cpu["cpu0"][mem_addr]["write_block_set"]
            cpu1_read_block_set = \
                    shared_mem_access_dict_by_cpu["cpu1"][mem_addr]["read_block_set"]
            for write_block_addr in cpu0_write_block_set:
                for read_block_addr in cpu1_read_block_set:
                    if write_block_addr == read_block_addr:
                        continue
                    flow = Flow(write_block_addr, read_block_addr, "cpu0", "cpu1")
                    inter_data_flow.add(flow)
            cpu1_write_block_set = \
                    shared_mem_access_dict_by_cpu["cpu1"][mem_addr]["write_block_set"]
            cpu0_read_block_set = \
                    shared_mem_access_dict_by_cpu["cpu0"][mem_addr]["read_block_set"]
            for write_block_addr in cpu1_write_block_set:
                for read_block_addr in cpu0_read_block_set:
                    if write_block_addr == read_block_addr:
                        continue
                    flow = Flow(write_block_addr, read_block_addr, "cpu1", "cpu0")
                    inter_data_flow.add(flow)
        return inter_data_flow

    def get_shortcut_flow(self, sti, shortcut_edge_hop):
        """ Add shortcut edges that connect every k-th blocks in the sc code flow """
        pickle_filepath = osp.join(self.sti_data_dirpath, "code_block_sequence", \
                sti.id, sti.cpu)
        if osp.exists(pickle_filepath) is False:
            return None
        with open(pickle_filepath, "rb") as pickle_file:
            code_block_sequence = pickle.load(pickle_file)

        shortcut_flow = set([])
        cpu = sti.cpu
        block_sequence_len = len(code_block_sequence)
        prev_pos = None
        for curr_pos in range(0, block_sequence_len, shortcut_edge_hop):
            curr_block_addr = code_block_sequence[curr_pos]
            if prev_pos is not None:
                prev_block_addr = code_block_sequence[prev_pos]
                shortcut_flow.add(Flow(prev_block_addr, curr_block_addr, \
                        cpu, cpu))
            prev_pos = curr_pos
        return shortcut_flow

    def get_schedule_flow(self, ct):
        """ Build the re-schedule flows based on the scheduling hint """
        # TODO: improve the doc/explanation of this method

        schedule = ct.schedule
        init_cpu = schedule.init_cpu
        if init_cpu == "cpu0":
            sti = ct.cti.sti_by_cpu["cpu1"]
        else:
            sti = ct.cti.sti_by_cpu["cpu0"]
        pickle_filepath = osp.join(self.sti_data_dirpath, "code_block_sequence", \
                sti.id, sti.cpu)
        if osp.exists(pickle_filepath) is False:
            return
        with open(pickle_filepath, "rb") as pickle_file:
            code_block_sequence = pickle.load(pickle_file)
        second_cpu_first_block_addr = code_block_sequence[0]

        schedule_flow = set([])
        if init_cpu == "cpu0":
            flow = Flow(schedule.cpu0_switch_point, second_cpu_first_block_addr, \
                    "cpu0", "cpu1")
            schedule_flow.add(flow)
            flow = Flow(schedule.cpu1_switch_point, schedule.cpu0_switch_point, \
                    "cpu1", "cpu0")
            schedule_flow.add(flow)
        elif init_cpu == "cpu1":
            flow = Flow(schedule.cpu1_switch_point, second_cpu_first_block_addr, \
                    "cpu1", "cpu0")
            schedule_flow.add(flow)
            flow = Flow(schedule.cpu0_switch_point, schedule.cpu1_switch_point, \
                    "cpu0", "cpu1")
            schedule_flow.add(flow)
        return schedule_flow
