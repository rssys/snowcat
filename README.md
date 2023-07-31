This repository contains the artifact for the SOSP'23 paper:

*Sishuai Gong, Dinglan Peng, Deniz Altınbüken, Pedro Fonseca, Petros Maniatis, "Snowcat: Efficient Kernel Concurrency Testing using a Learned Coverage Predictor".*



## 💡 Dear artifact reviewers

- The root README.md (this doc) provides instructions to help you verify Snowcat is **runnable**. Starting from scratch, we will first create a tiny dataset, train the PIC model based on the dataset and then use the PIC model to predict some concurrent tests.
- The [`artifact_evaluation` doc](doc/artifact_evaluation.md) provides instructions to help you verify the major results. In particular, they explain how to access the raw experiment data we collect and to verify the presentations (e.g., graphs in the paper).



## Overview of Snowcat implementation

The implementation of Snowcat has two major components---`learning` and `testing`.
- The `learning` component (code under `snowcat/learning/`)  is used to:
  - train the PIC model.
  - make inference using a trained PIC.
- The `testing` component (code under XXX) is used to:
  - collect training data.




## Getting started

### Prerequisite

- Minimum:
  
  OS: Ubuntu-18.04
  
  Processor: 32-core CPU
  
  Memory: 128GB memory
  
  Graphics: None
  
  Storage: 2TB disk + Btrfs compression enabled (see how to enable compression in this [doc](doc/btrfs_compression.md))

- Recommended:

  OS: Ubuntu-18.04

  Processor: 96-core CPU

  Memory: 680 memory

  Graphics: 8 Nvidia A100 (40GB) GPUs

  Storage: 20TB disk + Btrfs compression enabled (see how to enable compression in this [doc](doc/btrfs_compression.md))

**The following instructions assume a minimum hardware spec (no GPUs).**

### Get Snowcat ready (ETA: 30 minutes)

- The `testing` component requires some compilation and it can be done with the following commands:

  ```bash
  $ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
  $ cd script
  # source testing_setup.sh $SNOWCAT_STORAGE
  $ source testing_setup.sh snowcat-data # the path to store all outputs.
  ```

  Note: the above script exports several important ENV variables that are cruical for Snowcat. Please **ensure** the script is always sourced in your current bash session.

- The `learning` component requires installing some ML liraries. Please first **manually install anaconda** and then run the following commands:

  ```bash
  $ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
  $ cd script
  $ ./learning_setup_cpu.sh
  ```

  Note: A new Conda environment `snowcat-cpu` is created after running the above commands. Before moving next, it is **recommended** to exit the current bash session, start a new one and make sure the environment `snowcat-cpu` is activated. Also, please source `testing_setup.sh` again 😉.



## Step-1: Collect training data


### Collect the execution data of sequential test inputs (ETA: 15 minutes)
Some execution information about the sequential test inputs (e.g., control flow) is needed for Snowcat to generate the input graph to the model.

**How to run?**
This repo provides a coprus of sequential test inputs and they can be analyzed by Snowcat with the following commands:

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ cd script/data-collection/sti-data
# $ ./collect_sti_data.sh $STI_ID_A $STI_ID_B
$ ./collect_sti_data.sh 100 150
```
`$STI_ID_A` and `$STI_ID_B` together decide a range of sequential test inputs that will be analyzed. In the above case, 50 sequential test inputs (STI-ID==100, STI-ID=101, STI-ID=102, ..., STI-ID=150) will be executed and profiled.

**What output is expected?**

A folder `sti-data` under `$SNOWCAT_STORAGE` will be created.

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ ls $SNOWCAT_STORAGE/sti-data
block_assembly_dict  code_block_sequence  error  generator-log  intra_data_flow  raw  sc_control_flow  shared_mem_access  stat  ur_control_flow
```

### Collect the execution data of concurrent test inputs (ETA: 90 minutes)

The coverage of some concurrent test inputs is needed to build a training dataset so that Snowcat can learn from it.

**How to run?**

Running the following commands will collect the coverage of certain concurrent tests:

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ cd script/data-collection/cti-data
# $ ./collect_random_cti_data.sh $number_of_ctis_to_profile; ./extract_coverage.py
$ ./collect_ranom_cti_data.sh 50; ./extract_coverage.py  # collect the coverage of 50 concurrent test inputs under different interleavings
```

**What output is expected?**

A folder `cti-data` under `$SNOWCAT_STORAGE` will be created. Subfolder `dataset` is the dataset we built and `raw` is the place where raw execution traces are stored.

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ ls $SNOWCAT_STORAGE/cti-data
dataset  raw
```



## Step-2: Train a PIC model

💡For users who want to use the GPU (A100) for training:

Please modify the file `$ARTIFACT_HOME/learning/train_config_template.init` and `$ARTIFACT_HOME/learning/predict_config_template.init` by changing `use_cpu=False` to `use_cpu=True` before continue.

### Split the dataset (ETA: 5 minutes)

**How to run?**

To split the dataset (collected in the last step), one can run:

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ cd learning/
$ python split_dataset.py
```

**What output is expected?**

The script will tell the user where to find the dataset split and, more importantly, config files that can help you start training and inference immediately:

```python
the new dataset split is stored in ./dataset_split/split-2023-07-27-21-29-54
the new training config is stored in ./train-config-2023-07-27-21-29-54.ini
the new inference/predict config is stored in ./predict-config-2023-07-27-21-29-54.ini
```

(Note some error messages might show up to the screen but they are generally acceptable. Sometimes we might fail to generate a graph if certain information is missing (e.g., sequential control flow) and this issue occasionally happens when the random sequential test input is ill-defined.

### Start training

**How to run?**

Copy the path of the training config and run the following commands:

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ cd learning/
$ python train.py ./train-config-2023-07-27-20-54-32.ini
```

**What output is expected?**

1. In the middle of each epoch, a report of training or validation performance (average precision) is reported periocally.

   ```bash
   2023-07-27 23:31:52.923280 epoch: 01 loss: 0.07475609332323074 rank: 00 update_frequency: 2 trained_graphs_this_gpu: 100 total_graphs_per_gpu: 1238 train_ap_on_all: 0.9034862518310547 train_ap_on_ur: 0.01715177111327648 last_lr_rate: [5e-05]
   2023-07-28 01:03:21.298420 epoch: 01 loss: 0.04783577099442482 rank: 01 update_frequency: 2 trained_graphs_this_gpu: 200 total_graphs_per_gpu: 1238 train_ap_on_all: 0.9459865093231201 train_ap_on_ur: 0.013260450214147568 last_lr_rate: [5e-05]
   ```

2. After each epoch, a model checkpoint is created and save to the disk. One can find them in a folder under `$SNOWCAT_STORAGE/training/`. The folder is named as `train-{timestamp}` in which the timestamp is created at the beginning of the training.

   ```bash
   $ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
   $ ls $SNOWCAT_STORAGE/training
   train-2023-07-27-21-38-47/
   $ ls $SNOWCAT_STORAGE/training/train-2023-07-27-21-38-47/
   amp-checkpoint-0.tar  amp-checkpoint-1.tar  backup  bert-parameters  dataset-report  model-arch
   # amp-checkpoint-1.tar is the checkpoint made after the first epoch
   ```

   

## Step-3: Use the PIC model to make inference (ETA: 30 minutes)

**How to run?**

Once the model is trained, we can load the checkpoint and make inference on the test dataset. To start inference, one needs to:

1. Copy the path of the inference/predict config filepath, which is generated in the previous step: Split the dataset.
2. Copy the path of the model checkpoint, which can be found under the folder $SNOWCAT_STORAGE/training/, which is explained in the last step.

Then, running the following commands will make inference on the test dataset:

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ cd learning/
$ python predict.py ./predict-config-2023-07-27-21-29-54.ini $SNOWCAT_STORAGE/training/train-2023-07-27-21-38-47/amp-checkpoint-1.tar
```

**What output is expected?**

A folder under `$SNOWCAT_STORAGE/inference/` will be created and is named as `inference-{timestamp}`.

```bash
$ cd $ARTIFACT_HOME # the path where Snowcat is downloaded/cloned.
$ ls $SNOWCAT_STORAGE/inference
inference-2023-07-28-1-26-23/
```



## Step-4: Emulate SKI (MLPCT) (ETA: 3 minutes)
Based on the inference results of the test dataset, we can emulate a run of SKI.
Concurrent tests (CTs) that were predicted in the last step will be considered by different schedulers such as MLPCT and original PCT. In the end, each scheduler will select a few CTs that it wants to execute. Then, we can get the race coverage history achieved by this scheduler.

**How to run?**

Copy the path of the inference result `$SNOWCAT_STORAGE/inference/inference-{timestamp}` and run the following commands:

```bash
$ cd $ARTIFACT_HOME
$ cd evaluation/
$ python emulate_ski.py `$SNOWCAT_STORAGE/inference/inference-{timestamp}`
```

**What output is expected?**

A race coverage history graph `race-coverage-history.pdf` will be generated and stored under `$SNOWCAT_STORAGE/graph/`.

```bash
$ cd $SNOWCAT_STORAGE/graph
$ ls
race-coverage-history.pdf
```



## License

Code of Snowboard and SKI hypervisor (`tool/ski` and `tool/snowboard`) in this repository is licensed under the GLPv2 license (`tool/ski/src/LICENSE` and `tool/snowboard/src/LICENSE`). The rest of Snowcat implementation is under Apache 2 license (`script/LICENSE` and `learning/LICENSE`).
