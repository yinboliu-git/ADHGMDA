#!/bin/bash
# Script to create and configure the ADHGMDA virtual environment

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "Error: conda command not found. Please ensure Anaconda/Miniconda is properly installed and environment variables are configured."
    exit 1
fi

# Create virtual environment
echo "Creating ADHGMDA virtual environment (Python 3.8)..."
conda create -n ADHGMDA python=3.8 -y

# Activate virtual environment
echo "Activating ADHGMDA virtual environment..."
source $(conda info --base)/etc/profile.d/conda.sh  # Ensure conda command is available
conda activate ADHGMDA

# Check if environment activation was successful
if [ "$CONDA_DEFAULT_ENV" != "ADHGMDA" ]; then
    echo "Error: Failed to activate ADHGMDA virtual environment"
    exit 1
fi

# Install CUDA toolkit
echo "Installing cudatoolkit=11.7..."
conda install cudatoolkit=11.7 -c nvidia -y

# Install dgl
echo "Installing dgl==2.0.0+cu117..."
pip install dgl==2.0.0+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

# Install other dependencies
echo "Installing other required packages..."
pip install numpy pandas scikit-learn scipy joblib matplotlib seaborn networkx setproctitle pyyaml tqdm pyarrow

echo "All dependencies installed successfully! Currently activated environment: $CONDA_DEFAULT_ENV"
echo "To reactivate the environment later, use: conda activate ADHGMDA"
