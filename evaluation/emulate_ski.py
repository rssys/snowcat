import os
import os.path as osp
import sys
import multiprocessing
import pickle
import datetime
import traceback
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
# import custom modules
import inferenceUtils
import scheduler as SCHEDULER


BEST_THRESHOLD = 0.12


def create_log_dir(cti_inference_data_dirpath, result_dirpath):
    """Create a directory for logging."""
    cti_id = cti_inference_data_dirpath.split("/")[-1]
    log_dirpath = osp.join(result_dirpath, cti_id)
    os.makedirs(log_dirpath, exist_ok=True)
    return log_dirpath


def create_coverage_dict(cti_inference_data_dirpath, ct_id_list):
    """Generate a dict that returns the coverage given a ct_id."""
    global race_data_dirpath
    cti_id = cti_inference_data_dirpath.split("/")[-1]
    int(cti_id.split("_")[0])
    int(cti_id.split("_")[1])
    if osp.exists(osp.join(race_data_dirpath, cti_id, "schedule_info")) is True:
        coverage_data_dirpath = osp.join(race_data_dirpath, cti_id, "schedule_info")
    else:
        coverage_data_dirpath = osp.join(race_data_dirpath, cti_id)
    race_coverage_by_ct_id = {}
    for ct_id in ct_id_list:
        ct_dataset_dirpath = osp.join(coverage_data_dirpath, ct_id)
        race_coverage_filepath = osp.join(ct_dataset_dirpath, "race-coverage")
        assert osp.exists(race_coverage_filepath) is True
        with open(race_coverage_filepath, "rb") as tmp_file:
            race_coverage = pickle.load(tmp_file)
            race_coverage_by_ct_id[ct_id] = race_coverage
    return race_coverage_by_ct_id


def init_scheduler():
    """ Init a list of schedulers to evaluate. """
    global STRATEGY_LIST
    scheduler_list = []
    pct = SCHEDULER.PCT("pct")
    scheduler_list.append(pct)
    for strategy_config in STRATEGY_LIST:
        strategy_id = strategy_config[0]
        if strategy_id == 3:
            trial_limit = strategy_config[1]
            scheduler_name = f"mlpct-trained-strategy-{strategy_id}-limit-{trial_limit}"
        else:
            scheduler_name = f"mlpct-trained-strategy-{strategy_id}"
        mlpct = SCHEDULER.MLPCT(scheduler_name, strategy_config, False)
        scheduler_list.append(mlpct)
    return scheduler_list


def save_ct_coverage(log_dirpath, ct_id, race_coverage, debug):
    log_dirpath = osp.join(log_dirpath, "coverage", ct_id)
    ready_filepath = osp.join(log_dirpath, "ready")
    if osp.exists(ready_filepath) is True:
        return
    os.makedirs(log_dirpath, exist_ok=True)
    dump_filepath = osp.join(log_dirpath, "race-coverage")
    with open(dump_filepath, "wb") as f:
        pickle.dump(race_coverage, f)
    if debug is True:
        dump_filepath = osp.join(log_dirpath, "race-coverage.readable")
        with open(dump_filepath, "w") as f:
            for point in race_coverage:
                print(point, file=f)
    with open(ready_filepath, "w") as f:
        print("ready", file=f)


def save_cti_coverage(log_dirpath, overall_coverage, scheduler_name, debug=False):
    save_filepath = osp.join(log_dirpath, f"overall_coverage.{scheduler_name}")
    with open(save_filepath, "wb") as tmp_file:
        pickle.dump(overall_coverage, tmp_file)


def log_evaluation_error(log_dirpath, error_msg=None):
    log_filepath = osp.join(log_dirpath, "error")
    with open(log_filepath, "w") as log_file:
        if error_msg is None:
            traceback.print_exc(file=log_file)
            traceback.print_exc()
        else:
            print(error_msg, file=log_file)


def get_ct_id_list(cti_inference_data_dirpath):
    ct_id_list = []
    ct_id_list_filepath = osp.join(cti_inference_data_dirpath, "ct_id_list")
    if osp.exists(ct_id_list_filepath) is True:
        with open(osp.join(cti_inference_data_dirpath, "ct_id_list"), "r") as tmp_file:
            for line in tmp_file:
                ct_id_list.append(line.strip())
        return ct_id_list
    for tmp_f in os.scandir(cti_inference_data_dirpath):
        if tmp_f.name.endswith(".npy") is False:
            continue
        ct_id = tmp_f.name.split(".")[0]
        ct_id_list.append(ct_id)
    random.shuffle(ct_id_list)
    with open(osp.join(cti_inference_data_dirpath, "ct_id_list"), "w") as tmp_file:
        for ct_id in ct_id_list:
            print(ct_id, file=tmp_file)
    return ct_id_list


def emulate_one_cti(cti_inference_data_dirpath, result_dirpath):
    """Emulate the execution results of SKIs that use different scheduler."""
    global EXEC_BUDGET
    global STRATEGY_LIST
    global BEST_THRESHOLD
    log_dirpath = create_log_dir(cti_inference_data_dirpath, result_dirpath)
    try:
        ct_id_list = get_ct_id_list(cti_inference_data_dirpath)
        race_coverage_by_ct_id = create_coverage_dict(cti_inference_data_dirpath, \
                ct_id_list)
        scheduler_list = init_scheduler()
        overall_coverage_by_scheduler_name = {}
        for scheduler in scheduler_list:
            overall_coverage_by_scheduler_name[scheduler.name] = set([])
        for ct_id in ct_id_list:
            ct_predtrue_filepath = osp.join(cti_inference_data_dirpath, f"{ct_id}.npy")
            pos_block_set = \
                    inferenceUtils.fetch_pos_block_index(ct_predtrue_filepath, BEST_THRESHOLD)
            for scheduler in scheduler_list:
                # A scheduler stops considering any new cts when
                # it runs out of the execution budget.
                if scheduler.num_selected_ct >= EXEC_BUDGET:
                    continue
                is_selected = \
                        scheduler.consider_schedule(ct_id, pos_block_set)
                if is_selected is True:
                    race_coverage = race_coverage_by_ct_id[ct_id]
                    overall_coverage_by_scheduler_name[scheduler.name] |= race_coverage
                    # Save the coverage for future analysis
                    #save_ct_coverage(log_dirpath, ct_id, race_coverage, debug=True)
            should_break = True
            for scheduler in scheduler_list:
                if scheduler.num_selected_ct < EXEC_BUDGET:
                    should_break = False
            if should_break is True:
                break
        # Save decisions made by each scheduler
        for scheduler in scheduler_list:
            scheduler.save_all_decision(log_dirpath, debug=True)
            scheduler_name = scheduler.name
            overall_coverage = overall_coverage_by_scheduler_name[scheduler_name]
            save_cti_coverage(log_dirpath, overall_coverage, scheduler_name)
        print(f"Finished ski emulation on CTI {cti_inference_data_dirpath}")
    except:
        log_evaluation_error(log_dirpath)



def start_emulation(inference_data_dirpath, cti_id_list):
    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    assert snowcat_storage_dirpath is not None
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%Y-%m-%d-%H-%M-%S")
    result_dirpath = osp.join(snowcat_storage_dirpath, "evaluation", "ski", f"ski-emulation-{timestamp}")
    os.makedirs(result_dirpath, exist_ok=True)

    num_worker = int(multiprocessing.cpu_count() * 4)
    worker_pool = multiprocessing.Pool(num_worker)
    for cti_id in cti_id_list:
        cti_inference_data_dirpath = osp.join(inference_data_dirpath, cti_id)
        assert osp.exists(cti_inference_data_dirpath) is True
        worker_pool.apply_async(emulate_one_cti, \
                args = (cti_inference_data_dirpath, \
                        result_dirpath, ))
    worker_pool.close()
    worker_pool.join()
    return result_dirpath


class CoverageHistory():
    def __init__(self):
        self.current_coverage = set([])
        self.coverage_history = [0]

    def add_data(self, coverage):
        if coverage is not None:
            self.current_coverage |= coverage
        self.coverage_history.append(len(self.current_coverage))

    def export_history(self):
        return self.coverage_history


def get_graph_data(ski_result_dirpath, cti_id_list):
    scheduler_list = init_scheduler()
    scheduler_name_list = []
    coverage_history_by_scheduler_name = {}
    for scheduler in scheduler_list:
        scheduler_name_list.append(scheduler.name)
        coverage_history_by_scheduler_name[scheduler.name] = CoverageHistory()
    for cti_id in cti_id_list:
        log_dirpath = osp.join(ski_result_dirpath, cti_id)
        debug_file = open(osp.join(log_dirpath, "debug"), "w")
        msg_list = []
        msg_list.append(f"cti_id: {cti_id}")
        for scheduler_name in scheduler_name_list:
            overall_coverage_filepath = \
                    osp.join(log_dirpath, f"overall_coverage.{scheduler_name}")
            assert osp.exists(overall_coverage_filepath)
            with open(overall_coverage_filepath, "rb") as tmp_file:
                overall_coverage = pickle.load(tmp_file)
            coverage_tracker = coverage_history_by_scheduler_name[scheduler_name]
            msg_list.append(f"{scheduler_name}: {len(overall_coverage)}")
            coverage_tracker.add_data(overall_coverage)
        print(" ".join(msg_list), file=debug_file)
        debug_file.close() 
    graph_data = { \
            "time_exec" :  [], \
            "num_race" : [], \
            "scheduler" : [], \
            }
    for scheduler_name in coverage_history_by_scheduler_name:
        history = coverage_history_by_scheduler_name[scheduler_name].export_history()
        for num_executed_cti, coverage_size in enumerate(history):
            graph_data["time_exec"].append((num_executed_cti + 1) / 3 / 60)
            graph_data["num_race"].append(coverage_size)
            graph_data["scheduler"].append(scheduler_name)
    return pd.DataFrame.from_dict(graph_data)



def draw_graph(graph_data):
    high = 3.0
    width =  3.2

    fig, this_ax = plt.subplots(1, 1, figsize = (width, high))
    this_ax.clear()
    sns.lineplot(graph_data, x = "time_exec", y = "num_race", hue = "scheduler", ax = this_ax)
    this_ax.set_xlabel("Testing time (hours)")
    this_ax.set(ylabel=None)
    this_ax.set_xlim(0)

    plt.xlim(0)
    plt.grid()
    plt.grid(alpha=0.2)

    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    assert snowcat_storage_dirpath is not None
    save_dirpath = osp.join(snowcat_storage_dirpath, "graph")
    os.makedirs(save_dirpath, exist_ok = True)
    save_filepath = osp.join(save_dirpath, f"race-coverage-history.pdf")
    plt.tight_layout(pad = 0.0)
    plt.savefig(save_filepath)
    print(f"the graph is generated and stored at {save_filepath}")


def get_cti_id_list(inference_data_dirpath):
    cti_id_list = []
    cti_id_list_filepath = osp.join(inference_data_dirpath, "cti_id_list")
    if osp.exists(cti_id_list_filepath) is True:
        with open(cti_id_list_filepath, "r") as tmp_file:
            for line in tmp_file:
                cti_id_list.append(line.strip())
        return cti_id_list
    # Create a list from scratch
    for tmp_f in os.scandir(inference_data_dirpath):
        if tmp_f.is_dir() is False:
            continue
        try:
            int(tmp_f.name.split("_")[0])
            int(tmp_f.name.split("_")[1])
        except:
            continue
        cti_id_list.append(tmp_f.name)
    random.seed(1)
    random.shuffle(cti_id_list)
    assert osp.exists(cti_id_list_filepath) is False
    with open(cti_id_list_filepath, "w") as tmp_file:
        for cti_id in cti_id_list:
            print(cti_id, file=tmp_file)
    return cti_id_list

"""
3 strategies are proposed to select/discard schedules
based on the inference.
Each strategy can select `EXEC_BUDGET` number of schedules
to execute dynamically.
"""
EXEC_BUDGET=50
STRATEGY_LIST= []
STRATEGY_LIST.append([1, None])
STRATEGY_LIST.append([2, None])
STRATEGY_LIST.append([3, 3])
STRATEGY_LIST.append([3, 5])
STRATEGY_LIST.append([3, 8])


main_home = os.environ.get('MAIN_HOME')
if main_home is None:
    print("Please source testing_setup.sh")
    exit(1)

inference_data_dirpath = sys.argv[1]
if osp.exists(inference_data_dirpath) is False:
    print("inference data {inference_data_dirpath} does not exist")
    exit(0)
assert osp.exists(inference_data_dirpath) is True

race_data_dirpath = None
if len(sys.argv) == 3:
    race_data_dirpath = sys.argv[2]
else:
    snowcat_storage_dirpath = os.environ.get("SNOWCAT_STORAGE")
    race_data_dirpath = osp.join(snowcat_storage_dirpath, "cti-data", "dataset")
if osp.exists(race_data_dirpath) is False:
    print("race coverage data {race_data_dirpath} does not exist")
    exit(0)

cti_id_list = get_cti_id_list(inference_data_dirpath)
ski_result_dirpath = start_emulation(inference_data_dirpath, cti_id_list)
graph_data = get_graph_data(ski_result_dirpath, cti_id_list)
draw_graph(graph_data)
