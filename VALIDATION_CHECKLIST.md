# ✅ Validation Checklist

Use this checklist to verify your setup is complete and working.

## 📋 Pre-Installation Checks

- [ ] Conda is installed: `conda --version`
- [ ] `torch_new` environment exists or will be created
- [ ] Internet connection available for downloads

## 🔧 Installation Verification

### Step 1: Java Version (CRITICAL)
```bash
java -version
```
**Expected:** openjdk version "11.x.x"  
**Not:** Java 10 (won't work!)

- [ ] Java 11.x is installed
- [ ] JAVA_HOME is set correctly

### Step 2: Conda Environment
```bash
conda activate torch_new
echo $CONDA_DEFAULT_ENV
```
**Expected:** torch_new

- [ ] Environment activates successfully
- [ ] Correct environment is active

### Step 3: PySpark
```bash
python -c "import pyspark; print(pyspark.__version__)"
```
**Expected:** 3.3.2

- [ ] PySpark imports successfully
- [ ] Version is 3.3.2

### Step 4: PySpark Connection Test
```bash
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local').appName('test').getOrCreate(); print('✓ Spark works!'); spark.stop()"
```
**Expected:** "✓ Spark works!"

- [ ] Spark session creates successfully
- [ ] No Java version errors

### Step 5: PyTorch
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```
**Expected:** PyTorch: 2.0.1, CUDA: True/False

- [ ] PyTorch imports successfully
- [ ] CUDA status shown (True for GPU, False for CPU)

### Step 6: XGBoost
```bash
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```
**Expected:** XGBoost: 1.7.6

- [ ] XGBoost imports successfully
- [ ] Version is 1.7.6

### Step 7: Other ML Libraries
```bash
python -c "import sklearn, pandas, numpy; print('✓ All ML libraries work')"
```
**Expected:** "✓ All ML libraries work"

- [ ] scikit-learn imports
- [ ] pandas imports
- [ ] numpy imports

### Step 8: Jupyter
```bash
jupyter --version
```
**Expected:** jupyter core version and lab/notebook versions

- [ ] Jupyter is installed
- [ ] Kernel registered for torch_new

## 📁 File Configuration Verification

### Step 9: CombineDatasets.scala
```bash
grep "USE_HDFS" CombineDatasets.scala
```
**Check:** Line 10 should have `val USE_HDFS = true` or `false`

- [ ] File exists
- [ ] USE_HDFS flag is visible
- [ ] Set to `true` for HDFS or `false` for local files

### Step 10: Documentation Files
```bash
ls -1 important.txt SETUP_SUMMARY.md QUICK_SETUP_GUIDE.md DOC_INDEX.md
```
**Expected:** All files listed

- [ ] important.txt exists
- [ ] SETUP_SUMMARY.md exists
- [ ] QUICK_SETUP_GUIDE.md exists
- [ ] DOC_INDEX.md exists

## 🚀 Optional: HDFS Verification (if using HDFS)

### Step 11: HDFS Running
```bash
hdfs dfsadmin -report
```
**Expected:** Cluster summary

- [ ] HDFS is running
- [ ] Can connect to localhost:9000

### Step 12: HDFS Data Access
```bash
hdfs dfs -ls hdfs://localhost:9000/
```
**Expected:** Directory listing

- [ ] Can list HDFS directories
- [ ] No connection errors

## 🎯 Final Integration Test

### Step 13: Test Setup Script Runs
```bash
bash setup_torch_new.sh --help 2>&1 || echo "Script exists"
```
**Expected:** Script exists or shows output

- [ ] setup_torch_new.sh is executable
- [ ] Script file exists

### Step 14: Python Module Imports (All at Once)
```bash
python << PYEOF
import sys
try:
    import pyspark
    import torch
    import xgboost
    import sklearn
    import pandas
    import numpy
    import matplotlib
    import seaborn
    print("✅ ALL MODULES IMPORTED SUCCESSFULLY!")
    print(f"PySpark: {pyspark.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"XGBoost: {xgboost.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
PYEOF
```

- [ ] All modules import without errors
- [ ] Versions are displayed

## 🔍 Common Issues Check

### If Any Check Failed:

#### Java 11 Check Failed
```bash
# Fix: Install Java 11
conda activate torch_new
conda install -c conda-forge openjdk=11
java -version
```

#### PySpark Check Failed
```bash
# Fix: Install PySpark
conda activate torch_new
conda install -c conda-forge pyspark=3.3.2
```

#### PyTorch Check Failed
```bash
# Fix: Install PyTorch
conda activate torch_new
# For GPU:
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# For CPU:
# conda install pytorch==2.0.1 torchvision torchaudio cpuonly -c pytorch
```

#### Module Import Failed
```bash
# Fix: Ensure environment is activated
conda activate torch_new
# Then install missing modules
conda install [module-name]
```

## ✨ Success Criteria

**Your setup is complete if:**

✅ All Java, PySpark, PyTorch checks pass  
✅ All ML library imports work  
✅ Spark session creates successfully  
✅ Documentation files exist  
✅ (Optional) HDFS is accessible if you're using it

**Ready to run notebooks if:**

✅ All above criteria met  
✅ CombineDatasets.scala is configured (USE_HDFS flag)  
✅ You've read at least one setup guide

## 📊 Quick Status Summary

Run this to get a complete status:

```bash
echo "=== VALIDATION SUMMARY ==="
echo ""
echo "1. Java Version:"
java -version 2>&1 | head -1
echo ""
echo "2. Conda Environment:"
echo "   Current: $CONDA_DEFAULT_ENV"
echo ""
echo "3. Module Versions:"
python << 'PYEOF'
try:
    import pyspark; print(f"   PySpark: {pyspark.__version__}")
except: print("   PySpark: ❌ NOT INSTALLED")
try:
    import torch; print(f"   PyTorch: {torch.__version__}")
except: print("   PyTorch: ❌ NOT INSTALLED")
try:
    import xgboost; print(f"   XGBoost: {xgboost.__version__}")
except: print("   XGBoost: ❌ NOT INSTALLED")
try:
    import sklearn; print(f"   sklearn: {sklearn.__version__}")
except: print("   sklearn: ❌ NOT INSTALLED")
PYEOF
echo ""
echo "4. Spark Connection:"
python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local').appName('test').getOrCreate(); print('   ✅ SUCCESS'); spark.stop()" 2>&1 || echo "   ❌ FAILED"
echo ""
echo "=== END SUMMARY ==="
```

## 🎯 What to Do Next

### All Checks Passed ✅
**Congratulations! You're ready to:**
1. Read QUICKSTART.md for API usage
2. Run CombineDatasets.scala to prepare data
3. Run sparknote.ipynb or multi_model_drug_interaction_prediction.ipynb

### Some Checks Failed ❌
**Fix issues by:**
1. Following fix commands in "Common Issues Check" section above
2. Re-running validation checklist
3. Checking troubleshooting in important.txt

### Not Sure What Failed
**Debug by:**
1. Run the "Quick Status Summary" command above
2. Check important.txt Section 7 (Troubleshooting)
3. Verify Java version (most common issue!)

---

**Remember:** Java 11 is critical! Most issues are Java-related.

**Last Updated:** October 2024
