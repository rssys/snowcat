#!/usr/bin/python3
import os
import os.path as osp
import multiprocessing
import traceback
import sys
# import custom module(s)
import databuilder
main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
module_dirpath = osp.join(main_home, "learning/dataset/generator/")
sys.path.append(module_dirpath)
import testinput


def split_list(lst, num_chunks):
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    result = []
    start = 0
    for i in range(num_chunks):
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        result.append(lst[start:end])
        start = end
    return result

snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
result_dirpath = osp.join(snowcat_storage_dirpath, "sti-data")

error_dirpath = osp.join(result_dirpath, "error")
os.makedirs(error_dirpath, exist_ok = True)
log_dirpath = osp.join(result_dirpath, "generator-log")
os.makedirs(log_dirpath, exist_ok=True)

def worker_extract_sti_data(task_split):
    data_builder = databuilder.DataBuilder(result_dirpath)
    for task in task_split:
        sti = task[0]
        trace_filepath = task[1]
        try:
            data_builder.extract_sti_data(sti, trace_filepath)
            log_filepath = osp.join(log_dirpath, f"{sti.id}_{sti.cpu}")
            with open(log_filepath, "w") as f:
                print(f"{os.getpid()}", file=f)
        except:
            traceback.print_exc()
            error_filepath = osp.join(error_dirpath, f"{sti.id}_{sti.cpu}")
            with open(error_filepath, "w") as f:
                traceback.print_exc(file=f)

# find all tasks
profile_data_dirpath = sys.argv[1]
task_list = []
for cpu in ["cpu0", "cpu1"]:
    percpu_data_dirpath = osp.join(profile_data_dirpath, cpu)
    for f in os.scandir(percpu_data_dirpath):
        if f.is_dir() is False:
            continue
        curr_dirpath = osp.join(percpu_data_dirpath, f.name)
        for ff in os.scandir(curr_dirpath):
            if ff.is_dir() is False:
                continue
            sti_id = None
            try:
                sti_id_pair = ff.name.split("_")
                assert len(sti_id_pair) == 2
                if cpu == "cpu0":
                    sti_id = sti_id_pair[0]
                else:
                    sti_id = sti_id_pair[1]
            except:
                continue
            assert sti_id is not None
            sti = testinput.STI(sti_id, cpu)
            for fff in os.scandir(osp.join(curr_dirpath, ff.name)):
                if fff.name.startswith("trace") is True:
                    trace_filepath = osp.join(curr_dirpath, ff.name, fff.name)
                    task_list.append([sti, trace_filepath])

with open("log-task-list", "w") as f:
    for task in task_list:
        sti = task[0]
        trace_filepath = task[1]
        print(sti.id, sti.cpu, trace_filepath, file=f)

num_worker = int(multiprocessing.cpu_count() / 2)
worker_pool = multiprocessing.Pool(num_worker)
task_split_list = split_list(task_list, num_worker)
for task_split in task_split_list:
    worker_pool.apply_async(worker_extract_sti_data, args = (task_split, ))
worker_pool.close()
worker_pool.join()
