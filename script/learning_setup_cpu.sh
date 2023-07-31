#!/bin/bash
#
ENV_NAME="snowcat-cpu"
PYTHON_VERSION="3.9"

# Check if conda command is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Conda manually before running this script."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
    echo "Updating packages..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create --name "$ENV_NAME" python=$PYTHON_VERSION --yes
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
fi

conda install --yes pytorch=1.11.0 torchvision torchaudio cpuonly -c pytorch
conda install --yes torchmetrics
conda install --yes pyg=2.0.4 -c pyg
# Install fairseq from source code
git clone git@github.com:facebookresearch/fairseq.git
cd fairseq
git reset --hard 0f078de343d985e0cba6a5c1dc8a6394698
python -m pip install ./
cd ../
conda install --yes matplotlib seaborn

conda deactivate
echo "Conda environment '$ENV_NAME' created and packages installed."
echo "To activate the environment, use 'conda activate $ENV_NAME'."
