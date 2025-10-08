# Quick Setup Guide for torch_new Environment

## TL;DR - Quick Commands

```bash
# Step 1: Activate your conda environment
conda activate torch_new

# Step 2: Install Java 11 (CRITICAL - Java 10 won't work!)
conda install -c conda-forge openjdk=11 -y

# Step 3: Install PySpark
conda install -c conda-forge pyspark=3.3.2 -y

# Step 4: Install PyTorch (choose GPU or CPU)
# For GPU:
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# For CPU only:
# conda install pytorch==2.0.1 torchvision torchaudio cpuonly -c pytorch -y

# Step 5: Install ML libraries
conda install scikit-learn pandas numpy matplotlib seaborn -y
pip install xgboost==1.7.6

# Step 6: Install Jupyter
conda install jupyter notebook -y

# Step 7: Verify installations
java -version  # Should show Java 11
python -c "import pyspark; print(f'PySpark: {pyspark.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Why Java 11?

**Java 10 is NOT compatible with PySpark!**

PySpark requires:
- ✅ Java 8
- ✅ Java 11 (RECOMMENDED)
- ✅ Java 17
- ❌ Java 10 (NOT SUPPORTED)

## Verify Your Setup

```bash
# Check Java version (should be 11.x)
java -version

# Check PySpark
python -c "import pyspark; print(pyspark.__version__)"

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Test PySpark connection
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local').appName('test').getOrCreate(); print('✓ Spark works!'); spark.stop()"
```

## Common Issues

### Issue: "Unsupported class file major version"
**Fix:** Your Java version is wrong. Install Java 11:
```bash
conda activate torch_new
conda install -c conda-forge openjdk=11
```

### Issue: "Cannot import pyspark"
**Fix:** Install PySpark:
```bash
conda activate torch_new
conda install -c conda-forge pyspark=3.3.2
```

### Issue: "JAVA_HOME not set"
**Fix:** Set JAVA_HOME:
```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
# Add to ~/.bashrc to make permanent
echo 'export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))' >> ~/.bashrc
```

## What Each Notebook Needs

### `sparknote.ipynb` (PySpark MLlib)
Required:
- ✅ Java 11
- ✅ PySpark 3.3.2
- ✅ Python 3.8+
- Optional: HDFS (or modify code for local files)

### `multi_model_drug_interaction_prediction.ipynb` (PyTorch/XGBoost)
Required:
- ✅ PyTorch 2.0.1
- ✅ XGBoost 1.7.6
- ✅ scikit-learn
- ✅ pandas, numpy
- Optional: CUDA for GPU acceleration

### `CombineDatasets.scala` (Scala Spark)
Required:
- ✅ Java 11
- ✅ Scala 2.12.15
- ✅ SBT
- ✅ Apache Spark 3.3.2
- Optional: HDFS (or modify code for local files)

## Full Installation Script

Save this as `setup.sh` and run with `bash setup.sh`:

```bash
#!/bin/bash

# Quick setup script for torch_new environment
echo "Setting up torch_new environment..."

# Activate environment
conda activate torch_new

# Install Java 11
echo "Installing Java 11..."
conda install -c conda-forge openjdk=11 -y

# Install PySpark
echo "Installing PySpark..."
conda install -c conda-forge pyspark=3.3.2 -y

# Install PyTorch (GPU version - change to cpuonly if no GPU)
echo "Installing PyTorch..."
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ML libraries
echo "Installing ML libraries..."
conda install scikit-learn pandas numpy matplotlib seaborn -y
pip install xgboost==1.7.6

# Install Jupyter
echo "Installing Jupyter..."
conda install jupyter notebook -y

# Verify
echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "Verifying installations:"
java -version
echo ""
python -c "import pyspark; print(f'✓ PySpark: {pyspark.__version__}')"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import xgboost; print(f'✓ XGBoost: {xgboost.__version__}')"
echo ""
echo "Setup complete! You can now run the notebooks."
```

## Next Steps

1. ✅ Run the setup script above
2. ✅ Verify all installations
3. ✅ Read `important.txt` for detailed information
4. ✅ Run notebooks in this order:
   - First: `CombineDatasets.scala` (prepare data)
   - Then: `sparknote.ipynb` OR `multi_model_drug_interaction_prediction.ipynb`

## Need More Details?

See `important.txt` for:
- Complete version compatibility matrix
- HDFS setup instructions
- Troubleshooting guide
- System requirements
- Alternative configurations

## Quick Links

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [Java 11 Download](https://adoptium.net/temurin/releases/)
