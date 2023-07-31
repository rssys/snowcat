# Reproduce results of Snowcat

To facilitate the process for reviewers to reproduce our results, we make our raw experiment data publicly accessiable on Google Cloud Storage. Reviewers can download our raw data and run the script to reproduce the final results we present in the paper.



## Experiment data

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

  1. Please visit the website https://cloud.google.com/storage/docs/gsutil_install#linux to get gsutil installed in the machine.

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

One can selectively download the raw data to reproduce the a certain graph.

### Reproduce the graphs (ETA: 3 days downloading + 3 hours ski emulation)

**How to run?**

Taking figure-6.b as an example, one can reproduce the graph by running the following commands:

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
