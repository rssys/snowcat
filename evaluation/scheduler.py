import os
import pickle


class INTEREST_FN_1():
    def __init__(self):
        self.internal_state = set([])

    def is_unique(self, positive_possible_block_set):
        positive_possible_block_set = frozenset(positive_possible_block_set)
        if positive_possible_block_set not in self.internal_state:
            self.internal_state.add(positive_possible_block_set)
            return True
        return False


class INTEREST_FN_2():
    def __init__(self):
        self.internal_state = set([])

    def is_unique(self, positive_possible_block_set):
        if len(positive_possible_block_set - self.internal_state) > 0:
            self.internal_state |= positive_possible_block_set
            return True
        return False


class INTEREST_FN_3():
    def __init__(self, trial_limit):
        self.internal_state = {}
        self.trial_limit = trial_limit

    def is_unique(self, positive_possible_block_set):
        unique = False
        for possible_block_addr in positive_possible_block_set:
            if possible_block_addr not in self.internal_state:
                unique = True
            else:
                if self.internal_state[possible_block_addr] < self.trial_limit:
                    unique = True
            if unique is True:
                break
        for possible_block_addr in positive_possible_block_set:
            if possible_block_addr not in self.internal_state:
                self.internal_state[possible_block_addr] = 0
            self.internal_state[possible_block_addr] += 1
        return unique



class PCT():
    def __init__(self, name = "PCT"):
        self.name = name
        self.internal_state = None
        self.oracle = False
        self.history = []
        self.num_selected_ct = 0


    def consider_schedule(self, ct_id, possible_block_set):
        self.history.append([ct_id, True])
        self.num_selected_ct += 1
        return True


    def save_all_decision(self, log_dirpath, debug = False):
        log_filename = f"{self.name}.history"
        log_filepath = os.path.join(log_dirpath, log_filename)
        with open(log_filepath, "wb") as f:
            pickle.dump(self.history, f)
        if debug is False:
            return
        log_filename = f"{self.name}.history.readable"
        log_filepath = os.path.join(log_dirpath, log_filename)
        with open(log_filepath, "w") as f:
            for entry in self.history:
                print(entry[0], entry[1], file = f)
        log_filename = "overall"
        log_filepath = os.path.join(log_dirpath, log_filename)
        with open(log_filepath, "a") as f:
            num_selected = 0
            for entry in self.history:
                if entry[1] is True:
                    num_selected += 1
            print(f"{self.name} selected {num_selected} out-of {len(self.history)}", file = f)


def init_strategy(strategy_config):
    strategy_id = strategy_config[0]
    trial_limit = strategy_config[1]
    if strategy_id == 1:
        return INTEREST_FN_1()
    elif strategy_id == 2:
        return INTEREST_FN_2()
    elif strategy_id == 3:
        return INTEREST_FN_3(trial_limit)


class MLPCT(PCT):
    def __init__(self, name, strategy_config, oracle):
        super().__init__(name)
        self.strategy = init_strategy(strategy_config)
        self.oracle = oracle


    def consider_schedule(self, ct_id, possible_block_set):
        is_selected = self.strategy.is_unique(possible_block_set)
        if is_selected is True:
            self.num_selected_ct += 1
        self.history.append([ct_id, is_selected])
        return is_selected
