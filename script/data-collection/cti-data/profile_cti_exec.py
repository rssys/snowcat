#!/usr/bin/python3
import sys
import os
import time
import shutil
from datetime import datetime


"""Exercise different interleavings of the concurrent test input and
collect the execution trace for each interleaving"""
def data_collector(concurrent_input):
    main_home = os.environ.get('MAIN_HOME')
    if main_home is None:
        print("Please source testing_setup.sh")
        exit(1)
    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    assert snowcat_storage_dirpath is not None
    script = main_home + "/script/data-collection/cti-data/profile_cti_exec.sh"
    assert os.path.exists(script) is True

    log_file = open("trace-collector-log", "a")

    input1 = concurrent_input[0]
    input2 = concurrent_input[1]
    time_now = datetime.now()
    timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
    output_dirname = str(input1) + '_' + str(input2) + '_' + timestamp
    output_dirpath = os.path.join(snowcat_storage_dirpath, "cti-data", \
            "raw", output_dirname)
    os.makedirs(output_dirpath, exist_ok = True)

    cmd = ""
    cmd += "SKI_INPUT1_RANGE=%" + str(input1) + ' '
    cmd += "SKI_INPUT2_RANGE=%" + str(input2) + ' '
    cmd += "SKI_INTERLEAVING_RANGE=0-999 SKI_FORKALL_CONCURRENCY=30 "
    cmd += "SKI_OUTPUT_DIR=" + output_dirpath + ' ' + script
    
    # Start SKI to test this input
    print(cmd)
    ret = 0
    ret = os.system(cmd)

    # check if the execution finishes normally
    ret >>= 8
    if ret != 0:
        shutil.rmtree(output_dirpath, ignore_errors = True)
        print("Failed %d concurrent input %d %d %s" % (ret, input1, input2, timestamp), file = log_file)
        return

    time_now = datetime.now()
    timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")

    print("Finished concurrent input %d %d %s" % (input1, input2, timestamp), file = log_file)
    log_file.close()
