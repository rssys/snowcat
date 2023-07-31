import os

def extract_data_flow(trace, running_cpu):
    """ Extract the executed instruction sequence on a certain CPU """

    target_cpu_id = None
    assert running_cpu in ["cpu0", "cpu1"]
    if running_cpu == "cpu0":
        ins_pattern = "0"
        target_cpu_id = 0
    else:
        ins_pattern = "1"
        target_cpu_id = 1
    mem_access_pattern = f"### MEM: {target_cpu_id} c"

    data_flow_set = set([])
    shared_mem_access_dict = {}
    last_write_ins_dict = {}

    current_cr3 = 0
    target_cr3 = 0
    current_esp = 0
    int_handler_running = False
    prev_line = None
    recent_lines = []
    for line in trace:
        if len(recent_lines) < 20:
            recent_lines.append(line)
        else:
            recent_lines.pop(0)
            recent_lines.append(line)
            assert len(recent_lines) == 20
        # 1. Detect the execution of the interrupt handler
        if line.find("Executing interrupt handler") != -1:
            running_cpu_found = False
            check_line_idx = len(recent_lines) - 1
            while running_cpu_found is False and check_line_idx >= 0:
                check_line_idx -= 1
                cached_line = recent_lines[check_line_idx]
                if cached_line.find("Executing CPU") != -1:
                    line_split = cached_line.strip().split(" ")
                    cpu_id = int(line_split[3][ : 1])
                    if cpu_id == target_cpu_id:
                        int_handler_running = True
                    running_cpu_found = True
                elif cached_line.find("MEM") != -1:
                    line_split = cached_line.strip().split(" ")
                    cpu_id = int(line_split[2])
                    if cpu_id == target_cpu_id:
                        int_handler_running = True
                    running_cpu_found = True
                else:
                    try:
                        line_split = cached_line.strip().split(' ')
                        cpu_id = int(line_split[0])
                        if cpu_id == target_cpu_id:
                            int_handler_running = True
                        running_cpu_found = True
                    except:
                        continue
            assert running_cpu_found is True
        # 2. Detect the return of the interrupt handler
        if line.find("Returning from int") != -1:
            try:
                line_split = prev_line.strip().split(' ')
                cpu_id = int(line_split[0])
                if cpu_id == target_cpu_id:
                    int_handler_running = False
            except:
                print("==================")
                print(prev_line.strip())
                print("==================")
        # 3. Detect instructions executed (interrupt ignored)
        prev_line = line
        if int_handler_running is True:
            continue

        if line.startswith(ins_pattern) is True:
            line_split = line.strip().split(" ")
            if len(line_split) != 15:
                # broken lines, ignore
                continue
            current_cr3 = int(line_split[13], 16)
            if target_cr3 == 0:
                target_cr3 = current_cr3
            if current_cr3 == target_cr3:
                current_esp = int(line_split[6], 16)
        elif line.startswith(mem_access_pattern) is True:
            if current_cr3 != target_cr3:
                continue
            stack_begin = current_esp & 0xFFFFE000
            stack_end = stack_begin + 0x2000
            line_split = line.strip().split(" ")
            if len(line_split) != 10:
                # broken lines, ignore
                continue
            ins_addr = int(line_split[3], 16)
            mem_addr = int(line_split[4], 16)
            if stack_begin <= mem_addr < stack_end:
                continue
            if mem_addr not in shared_mem_access_dict:
                shared_mem_access_dict[mem_addr] = {"write_ins_set" : set([]), \
                        "read_ins_set" : set([])}
            access_type = line_split[8]
            if access_type == "S":
                last_write_ins_dict[mem_addr] = ins_addr
                shared_mem_access_dict[mem_addr]["write_ins_set"].add(ins_addr)
            else:
                assert access_type == "L"
                shared_mem_access_dict[mem_addr]["read_ins_set"].add(ins_addr)
                if mem_addr in last_write_ins_dict:
                    ins_pair = tuple([last_write_ins_dict[mem_addr], ins_addr])
                    data_flow_set.add(ins_pair)
    return data_flow_set, shared_mem_access_dict


def extract_ins_sequence(trace, running_cpu):
    """ Extract the executed instruction sequence on a certain CPU """

    target_cpu_id = None
    assert running_cpu in ["cpu0", "cpu1"]
    if running_cpu == "cpu0":
        ins_pattern = "0"
        target_cpu_id = 0
    else:
        ins_pattern = "1"
        target_cpu_id = 1

    ins_sequence = []
    current_cr3 = 0
    target_cr3 = 0
    int_handler_running = False
    prev_line = None
    recent_lines = []
    for line in trace:
        if len(recent_lines) < 20:
            recent_lines.append(line)
        else:
            recent_lines.pop(0)
            recent_lines.append(line)
            assert len(recent_lines) == 20
        # 1. Detect the execution of the interrupt handler
        if line.find("Executing interrupt handler") != -1:
            running_cpu_found = False
            check_line_idx = len(recent_lines) - 1
            while running_cpu_found is False and check_line_idx >= 0:
                check_line_idx -= 1
                cached_line = recent_lines[check_line_idx]
                if cached_line.find("Executing CPU") != -1:
                    line_split = cached_line.strip().split(" ")
                    cpu_id = int(line_split[3][ : 1])
                    if cpu_id == target_cpu_id:
                        int_handler_running = True
                    running_cpu_found = True
                elif cached_line.find("MEM") != -1:
                    line_split = cached_line.strip().split(" ")
                    cpu_id = int(line_split[2])
                    if cpu_id == target_cpu_id:
                        int_handler_running = True
                    running_cpu_found = True
                else:
                    try:
                        line_split = cached_line.strip().split(' ')
                        cpu_id = int(line_split[0])
                        if cpu_id == target_cpu_id:
                            int_handler_running = True
                        running_cpu_found = True
                    except:
                        continue
            assert running_cpu_found is True
        # 2. Detect the return of the interrupt handler
        if line.find("Returning from int") != -1:
            try:
                line_split = prev_line.strip().split(' ')
                cpu_id = int(line_split[0])
                if cpu_id == target_cpu_id:
                    int_handler_running = False
            except:
                print("==================")
                print(prev_line.strip())
                print("==================")
        # 3. Detect instructions executed (interrupt ignored)
        if line.startswith(ins_pattern) and int_handler_running is False:
            line_split = line.strip().split(' ')
            if len(line_split) != 15:
                # broken lines, ignore them for now
                continue
            ins = int(line_split[1], 16)
            current_cr3 = int(line_split[13], 16)
            if target_cr3 == 0:
                target_cr3 = current_cr3
            if target_cr3 == current_cr3:
                if ins >= 0xc1000000:
                    ins_sequence.append(ins)
        prev_line = line
    return ins_sequence
