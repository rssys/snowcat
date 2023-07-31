import os
import numpy as np


def load_ct_block_predtrue(ct_predtrue_filepath):
    ct_block_predtrue = np.load(ct_predtrue_filepath)
    return ct_block_predtrue


def fetch_pos_block_index(ct_predtrue_filepath, threshold):
    """Fetch the index of blocks that are predicted to be covered."""
    block_idx_set = set([])
    ct_block_predtrue = load_ct_block_predtrue(ct_predtrue_filepath)
    for block_idx in range(len(ct_block_predtrue)):
        node_prob_true = ct_block_predtrue[block_idx]
        prob = node_prob_true[0]
        if prob >= threshold:
            block_idx_set.add(block_idx)
    return block_idx_set
