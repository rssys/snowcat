# Collect cti data
To build a dataset for training and evaluating the PIC model,
one needs to know the code coverage when the concurrent test input
is executed under different interleavings. The dataset essentially
contains several examples where each example is a CT (test input + schedule)
and its corresponding coverage.

## How to collect cti data
Scripts in this folder provide two ways to collect coverage of CTs:
- `collect_random_cti_data.py` can generate random CTs and then coverage their coverage.

