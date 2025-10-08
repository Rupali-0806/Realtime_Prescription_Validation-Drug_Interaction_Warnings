#!/bin/bash

################################################################################
# Automated Setup Script for torch_new Environment
# Drug Interaction Prediction System
################################################################################

set -e  # Exit on any error

echo "================================================================================"
echo "Drug Interaction Prediction System - Automated Setup"
echo "================================================================================"
echo ""
echo "This script will install all required dependencies in your torch_new environment"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found!"
    echo "   Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found"

# Get environment name (default: torch_new)
ENV_NAME="${1:-torch_new}"
echo ""
echo "Target environment: $ENV_NAME"
echo ""

# Ask user for confirmation
read -p "Continue with installation? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "Starting installation..."
echo ""

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Environment '$ENV_NAME' exists"
else
    echo "⚠️  Environment '$ENV_NAME' not found!"
    read -p "Create new environment? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating environment '$ENV_NAME' with Python 3.10..."
        conda create -n $ENV_NAME python=3.10 -y
    else
        echo "Installation cancelled."
        exit 0
    fi
fi

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "❌ Error: Failed to activate environment '$ENV_NAME'"
    exit 1
fi

echo "✓ Environment '$ENV_NAME' activated"
echo ""

# Check current Java version
echo "Checking Java version..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
    echo "Current Java version: $JAVA_VERSION"
    if [[ "$JAVA_VERSION" == "10" ]]; then
        echo "⚠️  Warning: Java 10 is NOT compatible with PySpark!"
        echo "   Will install Java 11..."
    fi
else
    echo "Java not found. Will install Java 11..."
fi

# Install Java 11
echo ""
echo "Step 1/7: Installing Java 11..."
conda install -c conda-forge openjdk=11 -y
if [ $? -eq 0 ]; then
    echo "✓ Java 11 installed successfully"
    java -version
else
    echo "❌ Error installing Java 11"
    exit 1
fi

# Install PySpark
echo ""
echo "Step 2/7: Installing PySpark 3.3.2..."
conda install -c conda-forge pyspark=3.3.2 -y
if [ $? -eq 0 ]; then
    echo "✓ PySpark installed successfully"
    python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
else
    echo "❌ Error installing PySpark"
    exit 1
fi

# Ask about GPU support
echo ""
read -p "Do you have an NVIDIA GPU and want GPU acceleration? (y/N): " -n 1 -r
echo ""
GPU_SUPPORT=$REPLY

# Install PyTorch
echo ""
echo "Step 3/7: Installing PyTorch 2.0.1..."
if [[ $GPU_SUPPORT =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "Installing PyTorch CPU-only version..."
    conda install pytorch==2.0.1 torchvision torchaudio cpuonly -c pytorch -y
fi

if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed successfully"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
else
    echo "❌ Error installing PyTorch"
    exit 1
fi

# Install ML libraries
echo ""
echo "Step 4/7: Installing ML libraries..."
conda install scikit-learn=1.3.0 pandas=2.0.3 numpy=1.24.3 -y
if [ $? -eq 0 ]; then
    echo "✓ ML libraries installed successfully"
else
    echo "❌ Error installing ML libraries"
    exit 1
fi

# Install XGBoost
echo ""
echo "Step 5/7: Installing XGBoost..."
pip install xgboost==1.7.6
if [ $? -eq 0 ]; then
    echo "✓ XGBoost installed successfully"
    python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
else
    echo "❌ Error installing XGBoost"
    exit 1
fi

# Install visualization libraries
echo ""
echo "Step 6/7: Installing visualization libraries..."
conda install matplotlib seaborn plotly -y
if [ $? -eq 0 ]; then
    echo "✓ Visualization libraries installed successfully"
else
    echo "❌ Error installing visualization libraries"
    exit 1
fi

# Install Jupyter
echo ""
echo "Step 7/7: Installing Jupyter..."
conda install jupyter notebook ipykernel -y
if [ $? -eq 0 ]; then
    echo "✓ Jupyter installed successfully"
    # Register kernel
    python -m ipykernel install --user --name=$ENV_NAME --display-name="Python ($ENV_NAME)"
    echo "✓ Jupyter kernel registered"
else
    echo "❌ Error installing Jupyter"
    exit 1
fi

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install py4j==0.10.9.5
echo "✓ Additional dependencies installed"

# Summary
echo ""
echo "================================================================================"
echo "✅ Installation Complete!"
echo "================================================================================"
echo ""
echo "Installed versions:"
echo "-------------------"
java -version 2>&1 | head -n 1
python --version
python -c "import pyspark; print(f'PySpark: {pyspark.__version__}')" 2>/dev/null || echo "PySpark: Error"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Error"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')" 2>/dev/null || echo "XGBoost: Error"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "scikit-learn: Error"
echo ""

# Test PySpark connection
echo "Testing PySpark connection..."
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local').appName('test').getOrCreate(); print('✓ Spark connection successful!'); spark.stop()" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "⚠️  PySpark connection test failed. Please check the logs."
fi

echo ""
echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo ""
echo "1. Activate your environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Set JAVA_HOME (add to ~/.bashrc for persistence):"
echo "   export JAVA_HOME=\$(dirname \$(dirname \$(readlink -f \$(which java))))"
echo ""
echo "3. Start Jupyter Notebook:"
echo "   jupyter notebook"
echo ""
echo "4. Read the documentation:"
echo "   - SETUP_SUMMARY.md    - Overview of all files"
echo "   - QUICK_SETUP_GUIDE.md - Quick reference"
echo "   - important.txt       - Complete documentation"
echo ""
echo "5. Optional: Install HDFS (see important.txt Section 3)"
echo "   Or use local files (see SPARKNOTE_CONFIG_GUIDE.md)"
echo ""
echo "================================================================================"
echo "Happy coding! 🚀"
echo "================================================================================"
