import os
import os.path as osp
import pickle
import logging
import re
import sys
# import custom module(s)
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
module_dirpath = osp.join(main_home, "learning/dataset/generator/")
sys.path.append(module_dirpath)
import testinput
import kernelanalysis


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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


class DataBuilder():
    """ Extract data necessary for graph generation from the sti exec trace """
    def __init__(self, sti_data_dirpath):
        self.sti_data_dirpath = sti_data_dirpath
        kernel_info_dirpath = os.environ.get('KERNEL_INFO_DIR')
        if kernel_info_dirpath is None:
            print("Please source choose_kernel.sh")
            exit(1)
        self.kernel_info_dirpath = kernel_info_dirpath
        self.check_builder_prerequisite()

        # Generate the mapping from instruction addr to code block address
        self.ins_to_block_dict = {}
        self.block_range_list = []
        self.init_ins_to_block_dict()

        # Find the code block calling dependency
        self.block_calling_dict = None
        self.init_block_calling_dict()

        # Find the code assembly for each code block
        self.block_assembly_dict = {}
        self.init_block_assembly_dict()

    def check_builder_prerequisite(self):
        """ Assert the existance of certain static analysis results """
        assert osp.exists(self.kernel_info_dirpath) is True
        necessary_filename_list = ["vmlinux.map", "vmlinux.dis", \
                "block-info", "block-calling"]
        for necessary_filename in necessary_filename_list:
            necessary_filepath = osp.join(self.kernel_info_dirpath, necessary_filename)
            assert osp.exists(necessary_filepath) is True

    def init_ins_to_block_dict(self):
        """ Build a mapping from the ins address to the belonging block address """
        vmlinux_map_filepath = osp.join(self.kernel_info_dirpath, "vmlinux.map")
        with open(vmlinux_map_filepath, "r") as vmlinux_map_file:
            for line in vmlinux_map_file:
                ins_addr = int(line.strip()[ : 10], 16)
                self.ins_to_block_dict[ins_addr] = -1

        block_info_filepath = osp.join(self.kernel_info_dirpath, "block-info")
        with open(block_info_filepath, "r") as block_info_file:
            for line in block_info_file:
                line_split = line.strip().split(" ")
                block_start_addr = int(line_split[0], 16)
                block_end_addr = int(line_split[1], 16)
                block_range = tuple([block_start_addr, block_end_addr])
                self.block_range_list.append(block_range)
                for ins_addr in range(block_start_addr, block_end_addr):
                    if ins_addr not in self.ins_to_block_dict:
                        continue
                    self.ins_to_block_dict[ins_addr] = block_start_addr
        num_block = len(self.block_range_list)
        num_ins = len(self.ins_to_block_dict.keys())
        logger.debug(f"Mapped {num_ins} instructions to {num_block} blocks")

    def init_block_calling_dict(self):
        """ Load the code block calling dict """
        block_calling_info_filepath = osp.join(self.kernel_info_dirpath, "block-calling")
        with open(block_calling_info_filepath, "rb") as block_calling_info_file:
            self.block_calling_dict = pickle.load(block_calling_info_file)
        num_caller_block = len(self.block_calling_dict.keys())
        logger.debug(f"Found callees for {num_caller_block} blocks")

    def init_block_assembly_dict(self):
        """ Load the code assembly for each kernel instruction """
        ins_assembly_dict = {}
        vmlinux_dis_filepath = osp.join(self.kernel_info_dirpath, "vmlinux.dis")
        with open(vmlinux_dis_filepath, "r") as vmlinux_dis_file:
            for line in vmlinux_dis_file:
                if len(line) < 9 or line[8] != ':':
                    continue
                try:
                    ins_addr = int(line[:8], 16)
                except:
                    print("error happens when reading", line.strip())
                    continue
                ins_assembly = line[32:].strip()
                ins_assembly = re.sub(" +", " ", ins_assembly)
                ins_assembly_dict[ins_addr] = ins_assembly

        # Load the code assembly for each code block
        self.block_assembly_dict = {}
        for block_range in self.block_range_list:
            block_start_addr = block_range[0]
            block_end_addr = block_range[1]
            block_assembly = ""
            for ins_addr in range(block_start_addr, block_end_addr):
                if ins_addr in ins_assembly_dict:
                    block_assembly += ins_assembly_dict[ins_addr] + " ; "
            if block_assembly != "":
                self.block_assembly_dict[block_start_addr] = block_assembly
        num_block_assembly = len(self.block_assembly_dict.keys())
        logger.debug(f"Loaded code assembly for {num_block_assembly} blocks")

    def convert_to_block_sequence(self, ins_sequence):
        """ Convert the code block sequence given the instruction sequence """
        block_sequence = []
        prev_block_addr = -1
        for ins_addr in ins_sequence:
            if ins_addr not in self.ins_to_block_dict:
                continue
            block_addr = self.ins_to_block_dict[ins_addr]
            if block_addr == -1:
                continue
            if block_addr != prev_block_addr:
                block_sequence.append(block_addr)
                prev_block_addr = block_addr
        return block_sequence

    def get_sc_control_flow(self, code_block_sequence, cpu):
        """ Extract the sc control flow based on the SC block sequence """
        sc_control_flow = set([])
        for curr_pos in range(len(code_block_sequence) - 1):
            next_pos = curr_pos + 1
            curr_block_addr = code_block_sequence[curr_pos]
            next_block_addr = code_block_sequence[next_pos]
            flow = Flow(curr_block_addr, next_block_addr, cpu, cpu)
            sc_control_flow.add(flow)
        return sc_control_flow

    def get_ur_control_flow_by_hop(self, code_block_sequence, cpu, hop=1):
        ur_control_flow_by_hop = {}
        for curr_hop in range(1, hop + 1):
            ur_control_flow_by_hop[curr_hop] = set([])
        for curr_pos in range(len(code_block_sequence) - 1):
            curr_block_addr = code_block_sequence[curr_pos]
            next_block_addr = code_block_sequence[curr_pos + 1]
            parent_block_addr_set = set([curr_block_addr])
            covered_child_block_addr = next_block_addr
            ur_block_addr_set_this_hop = set([])
            curr_hop = 1
            # analyze UR blocks for curr_block_addr
            while curr_hop < hop + 1:
                ur_block_addr_set_this_hop = set([])
                for parent_block_addr in parent_block_addr_set:
                    parent_block_assembly = self.block_assembly_dict[parent_block_addr]
                    # "ret" block don't have uncovered reachable blocks
                    if parent_block_assembly.find("ret") != -1:
                        continue
                    child_block_addr_set = self.block_calling_dict[parent_block_addr]
                    #if curr_hop == 1:
                    child_block_addr_set.discard(covered_child_block_addr)
                    for child_block_addr in child_block_addr_set:
                        if child_block_addr not in self.block_assembly_dict:
                            continue
                        flow = Flow(parent_block_addr, child_block_addr, cpu, cpu)
                        ur_control_flow_by_hop[curr_hop].add(flow)
                        ur_block_addr_set_this_hop.add(child_block_addr)
                parent_block_addr_set = ur_block_addr_set_this_hop
                curr_hop += 1
        return ur_control_flow_by_hop

    def get_ur_control_flow(self, code_block_sequence, cpu):
        """ Generate the ur control flow based on the SC block sequence """
        # TODO: support finding multiple-hops UR blocks
        ur_control_flow = set([])
        for curr_pos in range(len(code_block_sequence) - 1):
            curr_block_addr = code_block_sequence[curr_pos]
            curr_block_assembly = self.block_assembly_dict[curr_block_addr]
            # "ret" block don't have uncovered reachable blocks
            if curr_block_assembly.find("ret") != -1:
                continue
            next_pos = curr_pos + 1
            next_block_addr = code_block_sequence[next_pos]
            ur_block_addr_set = self.block_calling_dict[curr_block_addr]
            for ur_block_addr in ur_block_addr_set:
                assert isinstance(ur_block_addr, int)
                break
            ur_block_addr_set.discard(next_block_addr)
            for ur_block_addr in ur_block_addr_set:
                if ur_block_addr not in self.block_assembly_dict:
                    continue
                ur_control_flow.add(Flow(curr_block_addr, ur_block_addr, cpu, cpu))
        return ur_control_flow

    def get_intra_data_flow(self, intra_data_flow_ins_pair_list, cpu):
        """ Generate the intra thread data flow """
        intra_data_flow = set([])
        for ins_pair in intra_data_flow_ins_pair_list:
            src_ins_addr = ins_pair[0]
            dst_ins_addr = ins_pair[1]
            if src_ins_addr not in self.ins_to_block_dict or \
                    dst_ins_addr not in self.ins_to_block_dict:
                continue
            src_block_addr = self.ins_to_block_dict[src_ins_addr]
            dst_block_addr = self.ins_to_block_dict[dst_ins_addr]
            if src_block_addr == dst_block_addr:
                continue
            intra_data_flow.add(Flow(src_block_addr, dst_block_addr, cpu, cpu))
        return intra_data_flow

    def convert_shared_mem_access_dict(self, shared_mem_access_dict):
        """ Convert the instruction address to block address in the mem dict """
        new_shared_mem_access_dict = {}
        for mem_addr in shared_mem_access_dict:
            new_shared_mem_access_dict[mem_addr] = {"write_block_set": set([]), \
                    "read_block_set": set([])}
            for ins_addr in shared_mem_access_dict[mem_addr]["write_ins_set"]:
                if ins_addr not in self.ins_to_block_dict:
                    continue
                block_addr = self.ins_to_block_dict[ins_addr]
                new_shared_mem_access_dict[mem_addr]["write_block_set"].add(block_addr)
            for ins_addr in shared_mem_access_dict[mem_addr]["read_ins_set"]:
                if ins_addr not in self.ins_to_block_dict:
                    continue
                block_addr = self.ins_to_block_dict[ins_addr]
                new_shared_mem_access_dict[mem_addr]["read_block_set"].add(block_addr)
        return new_shared_mem_access_dict

    def _save_sti_data(self, sti, data, data_type):
        """ Save the sti data as pickle to the disk """
        save_filepath = osp.join(self.sti_data_dirpath, data_type, \
                sti.id, sti.cpu)
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        with open(save_filepath, "wb") as save_file:
            pickle.dump(data, save_file)

    def save_sti_data(self, sti, data_dict):
        """ Save all sti data to disk """
        stat_msg = ""
        for data_type in data_dict:
            data = data_dict[data_type]
            self._save_sti_data(sti, data, data_type)
            stat_msg += f"{data_type}_len: {len(data)} "
        stat_save_filepath = osp.join(self.sti_data_dirpath, "stat", \
                sti.id, sti.cpu)
        os.makedirs(osp.dirname(stat_save_filepath), exist_ok=True)
        with open(stat_save_filepath, "w") as save_file:
            print(stat_msg.strip(), file=save_file)

    def extract_block_assembly(self, sc_control_flow, ur_control_flow):
        """ Build a assembly dictionary for code blocks in the graph """
        block_addr_set = set([])
        for flow in sc_control_flow:
            block_addr_set.add(flow.src.block_addr)
            block_addr_set.add(flow.dst.block_addr)
        if isinstance(ur_control_flow, set):
            for flow in ur_control_flow:
                block_addr_set.add(flow.src.block_addr)
                block_addr_set.add(flow.dst.block_addr)
        else:
            for hop in ur_control_flow:
                ur_control_flow_this_hop = ur_control_flow[hop]
                for flow in ur_control_flow_this_hop:
                    block_addr_set.add(flow.src.block_addr)
                    block_addr_set.add(flow.dst.block_addr)
        block_assembly_dict = {}
        for block_addr in block_addr_set:
            block_assembly_dict[block_addr] = self.block_assembly_dict[block_addr]
        return block_assembly_dict

    def extract_sti_data(self, sti, sti_trace_filepath):
        """
        - sequentially covered control flow
        - uncovered reachable control flow
        - intra thread data flow
        - inter thread data flow
        - assembly of code blocks
        """
        debug_msg = f"sti_id: {sti.id} cpu: {sti.cpu} "
        data_dict = {}
        with open(sti_trace_filepath, "r") as trace:
            ins_sequence = kernelanalysis.extract_ins_sequence(trace, sti.cpu)
        #print(f"len(ins_sequence): {len(ins_sequence)}")
        debug_msg += f"len(ins_sequence): {len(ins_sequence)} "
        code_block_sequence = self.convert_to_block_sequence(ins_sequence)
        data_dict["code_block_sequence"] = code_block_sequence
        debug_msg += f"len(code_block_sequence): {len(code_block_sequence)} "

        debug_msg += f"len(code_block_sequence): {len(code_block_sequence)} "
        sc_control_flow = self.get_sc_control_flow(code_block_sequence, sti.cpu)
        debug_msg += f"len(sc_control_flow): {len(sc_control_flow)} "
        #ur_control_flow = self.get_ur_control_flow(code_block_sequence, sti.cpu)
        #debug_msg += f"len(ur_control_flow): {len(ur_control_flow)} "

        ur_control_flow_by_hop = self.get_ur_control_flow_by_hop(code_block_sequence, sti.cpu)
        #assert ur_control_flow_by_hop[1] == ur_control_flow
        for hop in ur_control_flow_by_hop:
            ur_control_flow_this_hop = ur_control_flow_by_hop[hop]
            debug_msg += f"ur_control_flow_at_{hop}: {len(ur_control_flow_this_hop)} "
        #debug_msg += f"len(ur_control_flow): {len(ur_control_flow)} "
        data_dict["sc_control_flow"] = sc_control_flow
        data_dict["ur_control_flow"] = ur_control_flow_by_hop

        block_assembly_dict = self.extract_block_assembly(sc_control_flow, ur_control_flow_by_hop)
        data_dict["block_assembly_dict"] = block_assembly_dict
        debug_msg += f"len(block_assembly_dict): {len(block_assembly_dict)} "

        with open(sti_trace_filepath, "r") as trace:
            intra_data_flow_ins_pair_list, shared_mem_access_dict = \
                    kernelanalysis.extract_data_flow(trace, sti.cpu)
        intra_data_flow = self.get_intra_data_flow(intra_data_flow_ins_pair_list, sti.cpu)
        shared_mem_access_dict = self.convert_shared_mem_access_dict(shared_mem_access_dict)
        data_dict["intra_data_flow"] = intra_data_flow
        debug_msg += f"len(intra_data_flow): {len(intra_data_flow)} "
        data_dict["shared_mem_access"] = shared_mem_access_dict
        debug_msg += f"len(shared_mem_access_dict): {len(shared_mem_access_dict)}"

        self.save_sti_data(sti, data_dict)
        print(debug_msg)
