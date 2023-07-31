# Reproduce results of Snowcat

To facilitate the process (e.g., time) for reviewers to reproduce our results, we make our raw experiment data publicly accessiable on Google Cloud Storage. Reviewers can download our raw data and run the script to reproduce figures and diargrams we present in the paper.

## MLPCT (Figure-6)

## Prerequisite

Reproducing each subfigure in Figure-6 requires downloading ~5TB raw data, so we recommend using a ??TB disk with btrfs compression enabled.

### Access the experiment data

The experiment data is stored in `????` and there are generally two ways to download them.

- Use gsutil to download the data (recommended)

  1. Please visit the website https://cloud.google.com/storage/docs/gsutil_install#linux to get gsutil installed in the working machine.

  2. Set up a google credential (explained in the website too). A random google account is needed so that `gsutil` can access public data on the cloud.

  3. Run the command:

     ```bash
     cd $SNOWCAT_STORAGE
     gsutil -m cp -r gs://???????? .
     ```

- Visit the website to download the data

### Reproduce the graphs

Once the raw data is downloaded, one can draw the graphs.

