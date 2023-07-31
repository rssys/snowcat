# Reproduce results of Snowcat

## Overview

This doc provides instructions to reproduce our experiment results:

- **MLPCT:** Race coverage history when using different versions of the PIC model to test the kernel.
- **Snowboard-PIC:** finding ability when using the PIC model to select concurrent tests from a cluster of tests.

We consider them as our major results because:

- The **MLPCT** experiment shows the benefit of using PIC to select interesting schedules and potential of fine tuning a base PIC model for testing newer kernel versions.
- The **Snowboard-PIC** shows that using PIC to select concurrent tests can increase the ability (i.e., bug finding probability) of Snowboard to find new bugs.

In addition, this document introduces (1) how to access the experiment raw data and (2) how to access the trained PIC models.



## Experiment data

To facilitate the process for reviewewrs to reproduce our results, we make our raw experiment data publicly accessiable on Google Cloud Storage. Reviewers can download our raw data and run the script to reproduce the final results we present in the paper.

Our raw experiment data is stored at https://console.cloud.google.com/storage/browser/snowcat_sosp_2023 and can be accessed by anyone who has a google account.

### Content

Data related to the following experiments are uploaded:

- Inference result for MLPCT/SKI-PIC.

  We used different models to predict new concurrent tests (CTs) and then apply the new MLPCT, which will access the inference result for a predict, to select/skip CTs. With the inference data, one can simluate SKI runs and reproduce the same race coverage history graphs as we present in the paper.

- Inference result for Snowboard-PIC.

  We used the PIC-6.b model to predict concurrent tests and apply the Snowboard-PIC sampling method to select interesting concurrent tests to run. With the inference data, one can simluate Snowboard runs, know what interesting concurrent tests were sampled, and reproduce the similar results we present in the paper.

- Trained PIC models.

  One can use our trained models to make new inference.

### Access the data

There are generally two ways to download them.

- Use gsutil to download the data (recommended)

  1. Visit the website https://cloud.google.com/storage/docs/gsutil_install#linux to get gsutil installed.

  2. Set up a google credential (explained in the website as well). A google account is needed so that `gsutil` can access public data on the cloud.

  3. Run the command:

     ```bash
     gsutil -m cp -r gs://snowcat_sosp_2023/??? .
     ```

- Visit the website to download the data

  One can download the data using their browser with a few button clicks.



## MLPCT (Figure-6)

### Prerequisite

Reproducing each subfigure in Figure-6 requires downloading ~5TB raw data, so we recommend using a 10 TB disk with btrfs compression enabled.

### Experiment data

All data relevant to this experiment is stored under `snowcat_sosp_2023/mlpct/`:

```bash
- figure-6.a/ # inference data to reproduce this figure
- figure-6.b/ # inference data to reproduce this figure
- figure-6.d/ # inference data to reproduce this figure
- figure-6.f/ # inference data to reproduce this figure
- kernel-5.12-race-coverage-result.tar # possible data races found by dynamically running CTs. Needed for figure-6.a.
- kernel-6.1-race-coverage-result.tar # possible data races found by dynamically running CTs. Needed for figure-6.b, 6.d, 6.f.
```

(One can selectively download the raw data to reproduce the a certain graph.)

### Reproduce the graphs (ETA: 3 days downloading + 3 hours ski emulation)

**How to run?**

Taking figure-6.b as an example, one can reproduce this graph by running the following commands:

```bash
$ cd $SNOWCAT_STORAGE
$ gcloud storage cp -r gs://snowcat_sosp_2023/mlpct/kernel-6.1-race-coverage-result.tar . # this takes some time
$ tar -xvf kernel-6.1-race-coverage-result.tar
$ gcloud storage cp -r gs://snowcat_sosp_2023/mlpct/figure-6.b . # this takes some time
$ cd figure-6.b
$ tar -xvf inference-result.tar
# all necessary data is downloaded

$ cd $MAIN_HOME/evaluation
$ python emulate_ski.py $SNOWCAT_STORAGE/figure-6.b/inference-result $SNOWCAT_STORAGE/kernel-6.1-race-coverage-result
```

**What output is expected?**

A graph will be generated under `$SNOWCAT_STORAGE/graph` and it should be the same as the graph we present in the paper.




## Snowboard-PIC (Table-4)

### Experiment data

The data relevant to this experiment is stored under `snowcat_sosp_2023/snowboard-pic/`:

```bash
- bug-inference-data.tar
```

### Reproduce the table data (ETA: 15 minutes)

**How to run?**

One can reproduce the table numbers by running the following commands:

```bash
$ cd $SNOWCAT_STORAGE
$ gcloud storage cp -r gs://snowcat_sosp_2023/snowboard-pic/bug-inference-data.tar . # this takes some time
$ tar -xvf bug-inference-data.tar
$ cd bug-inference-data/
$ ls
bug-1 bug-2 bug-3 bug-4 bug-5 bug-6
# all necessary data is downloaded

$ cd $MAIN_HOME/evaluation/
$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-1
```

**What output is expected?**

```bash
$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-1
Bug finding probability:
Method: SB-PIC-S2 Probability: 1.0
Method: SB-RAND-S2 Probability: 0.538
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 29
Method: SB-RAND-S2 #-selected-ctis-avg: 29

$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-2
Bug finding probability:
Method: SB-PIC-S2 Probability: 0.669
Method: SB-RAND-S2 Probability: 0.493
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 20
Method: SB-RAND-S2 #-selected-ctis-avg: 20

$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-3
Bug finding probability:
Method: SB-PIC-S2 Probability: 0.589
Method: SB-RAND-S2 Probability: 0.402
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 86
Method: SB-RAND-S2 #-selected-ctis-avg: 86

$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-4
Bug finding probability:
Method: SB-PIC-S2 Probability: 1.0
Method: SB-RAND-S2 Probability: 0.419
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 24
Method: SB-RAND-S2 #-selected-ctis-avg: 24

$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-5
Bug finding probability:
Method: SB-PIC-S2 Probability: 1.0
Method: SB-RAND-S2 Probability: 0.78
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 38
Method: SB-RAND-S2 #-selected-ctis-avg: 38

$ python emulate_snowboard.py $SNOWCAT_STORAGE/bug-infernece-data/bug-6
Bug finding probability:
Method: SB-PIC-S2 Probability: 0.51
Method: SB-RAND-S2 Probability: 0.195
Average number of selected ctis:
Method: SB-PIC-S2 #-selected-ctis-avg: 154
Method: SB-RAND-S2 #-selected-ctis-avg: 154
```



## Trained PIC models

### Experiment data
The evaluation of Snowcat used several versions of the PIC model, which are stored at `snowcat_sosp_2023/model-checkpoint/`:
```bash
pic-5-checkpoint.tar
pic-6.a-checkpoint.tar
pic-6.b-checkpoint.tar
pic-6.c-checkpoint.tar
pic-6.d-checkpoint.tar
```
One can download the checkpoint and use it to make new inference. Instructions related to inference are provided in the root README.
