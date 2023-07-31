#!/usr/bin/python3
import os
import sys
from datetime import datetime
import multiprocessing


'''
This script runs the profiling execution for seuqential inputs whose id is within range start ~ end
start_id sys.argv[1] type:int
starting sequential input test id
end_id sys.argv[2] type:int
ending sequential input test id
'''
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])
input_range = [start_id, end_id]
# create a timestamp
time_now = datetime.now()
timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")

main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)
snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
assert snowcat_storage_dirpath is not None

result_dirpath = os.path.join(snowcat_storage_dirpath, "sti-data", "raw", \
        f"profile-{timestamp}")
if not os.path.isdir(result_dirpath):
    os.makedirs(result_dirpath)
# Each sequential test will be profiled twice running on different CPUs
for cpu in range(0, 2):
    current_dirpath = os.path.join(result_dirpath, f"cpu{cpu}")
    if not os.path.isdir(current_dirpath):
        os.makedirs(current_dirpath)
    for input_index in range(input_range[0], input_range[1], 5000):
        cmd = ""
        if cpu == 0:
            cmd += "SKI_INPUT1_RANGE=" + str(input_index)+'-' + str(min(input_index+4999, input_range[1])) + ' '
            cmd += "SKI_INPUT2_RANGE=1-1 "
            cmd += "SKI_TRACE_SET_CPU=0 "
        elif cpu == 1:
            cmd += "SKI_INPUT2_RANGE=" + str(input_index)+'-' + str(min(input_index+4999, input_range[1])) + ' '
            cmd += "SKI_INPUT1_RANGE=1-1 "
            cmd += "SKI_TRACE_SET_CPU=1 "
        cmd += 'SKI_OUTPUT_DIR='+current_dirpath + ' '
        cmd += 'SKI_FORKALL_CONCURRENCY='+str(multiprocessing.cpu_count())
        cmd += ' ./profile_sti_exec.sh'
        print(cmd)
        ret = os.system(cmd)
        ret >>= 8
        if ret != 0:
            exit(1)
