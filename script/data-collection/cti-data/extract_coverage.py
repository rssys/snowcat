#!/usr/bin/python3
import os
import os.path as osp
import ast
import sys
import pickle
import multiprocessing
import traceback
import time
import signal


main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")

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


def convert_to_block_sequence(ins_sequence):
    """ Convert the code block sequence given the instruction sequence """
    global INS_TO_BLOCK_DICT
    assert len(INS_TO_BLOCK_DICT.keys()) > 0
    block_sequence = []
    prev_block_addr = -1
    for ins_addr in ins_sequence:
        if ins_addr not in INS_TO_BLOCK_DICT:
            continue
        block_addr = INS_TO_BLOCK_DICT[ins_addr]
        if block_addr == -1:
            continue
        if block_addr != prev_block_addr:
            block_sequence.append(block_addr)
            prev_block_addr = block_addr
    return block_sequence


def get_cpu_ins_sequence(concurrent_trace):
    """ Get the instruction sequence on CPU-0 and CPU-1. """
    current_cpu = -1
    is_in_loop = False
    ins_sequence_by_cpu = {"cpu0" : [], "cpu1" : []}
    for line in concurrent_trace:
        line = line.strip()
        if line.find("###") != -1:
            # SKI rescheduled the thread
            if line.find("Executing CPU") != -1:
                current_cpu = int(line[line.find("Executing CPU") + 14])
                
            # SKI prints loop information
            if line.find("Loop start") != -1:
                assert is_in_loop is False
                is_in_loop = True

            if line.find("Loop end") != -1:
                assert is_in_loop is True
                is_in_loop = False

        else:
            fields = line.strip().split(' ')
            if len(fields) == 15:
                # Ignore instructions in loop
                if is_in_loop is True:
                    continue
                # Otherwise, the entry is in loop_tail_end (check SKI)

            # Double check the running CPU
            assert int(fields[0]) == current_cpu

            # Note: ignore instructions from other CPUs
            if current_cpu != 0 and current_cpu != 1:
                continue

            ins_addr = int(fields[1], 16)
            # Note: ignore userspace instructions
            if ins_addr < 0xc000000:
                continue

            if current_cpu == 0:
                cpu_str = "cpu0"
            else:
                cpu_str = "cpu1"
            ins_sequence_by_cpu[cpu_str].append(ins_addr)
    #print("Number of instructions on CPU-0 %d CPU-1 %d" % (len(ins_sequence_by_cpu["cpu0"]), len(ins_sequence_by_cpu["cpu1"]))) 
    return ins_sequence_by_cpu


def get_block_coverage(concurrent_exec_trace):
    """Extract the block coverage on CPU-0 and CPU-1 given the exec trace"""
    block_coverage_by_cpu = {}
    ins_sequence_by_cpu = get_cpu_ins_sequence(concurrent_exec_trace)
    for cpu in ["cpu0", "cpu1"]:
        ins_sequence = ins_sequence_by_cpu[cpu]
        block_sequence = convert_to_block_sequence(ins_sequence)

        block_coverage = set(block_sequence)
        block_coverage_by_cpu[cpu] = block_coverage
    return block_coverage_by_cpu


def convert_exec_para_to_dict(exec_para_line):
    """Convert a SKI parameter line into a dict."""
    exec_para_dict = {}
    fields = exec_para_line.split(' ')
    num_fields = len(fields)
    assert num_fields == 43

    field_idx = 0
    while field_idx < num_fields:
        assert fields[field_idx].find(":")
        key = fields[field_idx][ : -1]
        if key == "Exit":
            value = fields[field_idx + 1] + ' ' + fields[field_idx + 2]
            field_idx += 3
        else:
            value = fields[field_idx + 1]
            field_idx += 2
        exec_para_dict[key] = value

    return exec_para_dict


def get_initial_cpu(concurrent_trace_file):
    """ Return the index of CPU that is the first to run. """
    initial_cpu = -1
    current_cpu = -1
    for line in concurrent_trace_file:
        if line.find("###") != -1:
            # SKI rescheduled the thread
            if line.find("Executing CPU") != -1:
                current_cpu = int(line[line.find("Executing CPU") + 14])
                # If this is the initial CPU to run?
                if initial_cpu == -1:
                    initial_cpu = current_cpu
                    #print("initial cpu is %d" % (initial_cpu))
                    return initial_cpu


def get_schedule(trace_filepath, cpu0_preemption_ins, cpu1_preemption_ins):
    trace_file = open(trace_filepath, "r")
    initial_cpu = get_initial_cpu(trace_file)
    assert initial_cpu in [0, 1]
    schedule = tuple([str(initial_cpu), cpu0_preemption_ins, cpu1_preemption_ins])
    trace_file.close()
    return schedule


def get_race_coverage_dict(race_report_filepath):
    race_report = open(race_report_filepath, "r")
    race_coverage_by_tracename = {}
    for line in race_report:
        line_split = line.strip().split(" ")
        if len(line_split) != 32:
            continue
        tracename = line_split[1]
        ins1 = int(line_split[9], 16)
        ins2 = int(line_split[11], 16)
        if ins1 > ins2:
            race = tuple([ins2, ins1])
        else:
            race = tuple([ins1, ins2])
        if tracename not in race_coverage_by_tracename:
            race_coverage_by_tracename[tracename] = set([])
        race_coverage_by_tracename[tracename].add(race)
    race_report.close()
    return race_coverage_by_tracename


def get_file_folder_list(dirpath):
    file_list = []
    folder_list = []
    for f in os.scandir(dirpath):
        if f.is_dir() is True:
            folder_list.append(f.path)
        else:
            file_list.append(f.path)
    return file_list, folder_list


RESULT_DIRPATH = osp.join(snowcat_storage_dirpath, "cti-data", "dataset")
os.makedirs(RESULT_DIRPATH, exist_ok=True)
def collect_coverage_for_one_cti(concurrent_input_dirpath, sample_size=128):
    file_list, folder_list = get_file_folder_list(concurrent_input_dirpath)
    assert len(folder_list) == 1
    trace_folder_path = folder_list[0]

    # Search and download test parameter files
    exec_para_file_path = None
    race_report_filepath = None
    for file_path in file_list:
        if file_path.find("parameter") != -1:
            exec_para_file_path = file_path
        if file_path.find("race_detector") != -1:
            race_report_filepath = file_path
    assert exec_para_file_path is not None
    assert race_report_filepath is not None

    exec_para = []
    with open(exec_para_file_path, "r") as tmp_file:
        for line in tmp_file:
            exec_para.append(line.strip())
    # TODO: make this configurable, maybe use a env var?
    assert len(exec_para) == 1000

    race_coverage_by_tracename = get_race_coverage_dict(race_report_filepath)
    exec_para_dict = convert_exec_para_to_dict(exec_para[1])
    sti_id_cpu0 = exec_para_dict["I1"]
    sti_id_cpu1 = exec_para_dict["I2"]
    print(sti_id_cpu0, sti_id_cpu1)

    coverage_save_dirpath = osp.join(RESULT_DIRPATH, "_".join([sti_id_cpu0, sti_id_cpu1]))
    cmd = f"rm -r {coverage_save_dirpath}"
    os.system(cmd)
    os.makedirs(coverage_save_dirpath, exist_ok=True)
    schedule_info_dirpath = osp.join(coverage_save_dirpath, "schedule_info")
    os.makedirs(schedule_info_dirpath, exist_ok=True)

    mapping_filepath = osp.join(RESULT_DIRPATH, \
            "_".join([sti_id_cpu0, sti_id_cpu1]), \
            "_".join([sti_id_cpu0, sti_id_cpu1]) + ".mapping")
    mapping_file = open(mapping_filepath, "w")
    schedule_set = set([])

    for line in exec_para:
        exec_para_dict = convert_exec_para_to_dict(line)
        trace_filename = exec_para_dict["T"]

        #Get the graph info no matter if the trace is complete or not.
        #exit_reason = exec_para_dict["Exit"]
        #if exit_reason.find("HYPER_REQUEST") == -1:
        #    continue

        cpu0_preemption_ins = exec_para_dict["CPU0_preemption_ins"]
        cpu1_preemption_ins = exec_para_dict["CPU1_preemption_ins"]
        if cpu0_preemption_ins == "ffffffff" and cpu1_preemption_ins == "ffffffff":
            continue

        trace_filepath = os.path.join(trace_folder_path, trace_filename)
        try:
            schedule = get_schedule(trace_filepath, cpu0_preemption_ins, cpu1_preemption_ins)
        except:
            continue
        if schedule in schedule_set:
            continue

        schedule_save_dirname = "_".join([sti_id_cpu0, sti_id_cpu1, \
                schedule[0], schedule[1], schedule[2]])
        schedule_save_dirpath = os.path.join(RESULT_DIRPATH, \
                "_".join([sti_id_cpu0, sti_id_cpu1]), \
                "schedule_info", \
                schedule_save_dirname)

        try:
            concurrent_exec_trace = []
            with open(trace_filepath, "r") as tmp_file:
                for line in tmp_file:
                    concurrent_exec_trace.append(line.strip())
            block_coverage_by_cpu = get_block_coverage(concurrent_exec_trace)
            del concurrent_exec_trace

            print(schedule_save_dirname, trace_filepath, file = mapping_file)
            os.makedirs(schedule_save_dirpath, exist_ok=True)
            for cpu in block_coverage_by_cpu:
                save_filepath = osp.join(schedule_save_dirpath, f"{cpu}-coverage")
                with open(save_filepath, "wb") as f:
                    pickle.dump(block_coverage_by_cpu[cpu], f)

            # save the race coverage
            if trace_filename not in race_coverage_by_tracename:
                race_coverage = set([])
            else:
                race_coverage = race_coverage_by_tracename[trace_filename]
            save_filepath = osp.join(schedule_save_dirpath, "race-coverage")
            with open(save_filepath, "wb") as f:
                pickle.dump(race_coverage, f)
            save_filepath = osp.join(schedule_save_dirpath, "race-coverage.readable")
            with open(save_filepath, "w") as f:
                for race in race_coverage:
                    print(race, file=f)
            msg_split = []
        except:
            traceback.print_exc()
            continue

        for cpu in block_coverage_by_cpu:
            msg_split.append(f"{cpu}: {len(block_coverage_by_cpu[cpu])}")
        print(f"Finished {trace_filepath} {' '.join(msg_split)}")

        schedule_set.add(schedule)
        if len(schedule_set) == sample_size:
            break
    return


def run_one_worker(task):
    try:
        collect_coverage_for_one_cti(task)
    except:
        print("error-encountered", task)
        log_filepath = osp.join(ERROR_DIRPATH, f"worker-{os.getpid()}")
        with open(log_filepath, "a") as f:
            print("error-encountered", task, file=f)
            traceback.print_exc(file=f)


ERROR_DIRPATH = "node-coverage-collector-error"
os.makedirs(ERROR_DIRPATH, exist_ok=True)
def launch_worker_process(task_list):
    num_worker = multiprocessing.cpu_count()
    worker_pool = multiprocessing.Pool(num_worker)

    for task in task_list:
        worker_pool.apply_async(run_one_worker, args = (task, ))
    worker_pool.close()
    worker_pool.join()

if __name__ == "__main__":
    init_ins_to_block_dict()

    concurrent_trace_dirpath = osp.join(snowcat_storage_dirpath, \
            "cti-data", "raw")
    assert osp.exists(concurrent_trace_dirpath) is True
    # TODO: skip traces that are extracted in the last run?
    task_list = []
    for f in os.scandir(concurrent_trace_dirpath):
        if f.is_dir() is False:
            continue
        for ff in os.scandir(f.path):
            if ff.name.startswith("collect_data"):
                task_list.append(ff.path)
    print(task_list)
    launch_worker_process(task_list)
