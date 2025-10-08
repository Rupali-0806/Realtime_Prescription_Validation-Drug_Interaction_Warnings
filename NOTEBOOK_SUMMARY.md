# Sparknote.ipynb - Quick Reference Guide

## 📊 Overview
A PySpark MLlib notebook for distributed machine learning on drug interaction data from HDFS.

## 🎯 What It Does

```
┌─────────────────────────────────────────────────────────────┐
│                    SPARKNOTE.IPYNB                          │
│                                                             │
│  Input: HDFS Dataset                                        │
│  hdfs://localhost:9000/output/combined_dataset_complete.csv │
│                            ↓                                │
│         ┌──────────────────────────────────┐               │
│         │   Data Loading & Preprocessing    │               │
│         │  - Load from HDFS                 │               │
│         │  - Clean and encode labels        │               │
│         │  - Create numerical features      │               │
│         │  - Index drug columns             │               │
│         └──────────────────────────────────┘               │
│                            ↓                                │
│         ┌──────────────────────────────────┐               │
│         │     Train-Test Split (80/20)     │               │
│         └──────────────────────────────────┘               │
│                            ↓                                │
│    ┌────────────────────────────────────────────┐         │
│    │          Train 3 ML Models                 │         │
│    │  ┌──────────────────────────────────────┐ │         │
│    │  │ 1. Logistic Regression              │ │         │
│    │  │    - Regularization & Scaling        │ │         │
│    │  │    - Max Iterations: 100             │ │         │
│    │  └──────────────────────────────────────┘ │         │
│    │  ┌──────────────────────────────────────┐ │         │
│    │  │ 2. Random Forest                    │ │         │
│    │  │    - 100 Trees                       │ │         │
│    │  │    - Max Depth: 10                   │ │         │
│    │  └──────────────────────────────────────┘ │         │
│    │  ┌──────────────────────────────────────┐ │         │
│    │  │ 3. Gradient Boosted Trees           │ │         │
│    │  │    - 50 Iterations                   │ │         │
│    │  │    - Max Depth: 5                    │ │         │
│    │  └──────────────────────────────────────┘ │         │
│    └────────────────────────────────────────────┘         │
│                            ↓                                │
│         ┌──────────────────────────────────┐               │
│         │    Evaluate All Models           │               │
│         │  - Accuracy                       │               │
│         │  - Precision                      │               │
│         │  - Recall                         │               │
│         │  - F1-Score                       │               │
│         │  - ROC-AUC                        │               │
│         │  - PR-AUC                         │               │
│         └──────────────────────────────────┘               │
│                            ↓                                │
│         ┌──────────────────────────────────┐               │
│         │      Generate Visualizations     │               │
│         │  📊 Confusion Matrices           │               │
│         │  📈 ROC Curves                   │               │
│         │  📊 Metrics Comparison           │               │
│         │  📊 Feature Importance           │               │
│         └──────────────────────────────────┘               │
│                            ↓                                │
│         ┌──────────────────────────────────┐               │
│         │     Select & Save Best Model     │               │
│         │  (Based on ROC-AUC Score)        │               │
│         └──────────────────────────────────┘               │
│                            ↓                                │
│  Output: Trained Models + Visualizations                   │
│  - Best model saved to disk                                │
│  - 4 PNG visualization files                               │
│  - Complete metrics report                                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Ensure HDFS is running
start-dfs.sh

# Verify dataset exists
hdfs dfs -ls hdfs://localhost:9000/output/combined_dataset_complete.csv
```

### 2. Run Notebook
```bash
jupyter notebook sparknote.ipynb
```

### 3. Execute All Cells
Click "Cell" → "Run All" or press `Shift + Enter` for each cell

## 📦 Output Files

| File | Description | Type |
|------|-------------|------|
| `confusion_matrices.png` | 3 confusion matrices side-by-side | Image (300 DPI) |
| `roc_curves.png` | ROC curves for all models | Image (300 DPI) |
| `metrics_comparison.png` | Bar charts comparing metrics | Image (300 DPI) |
| `feature_importance.png` | Feature importance for RF & GBT | Image (300 DPI) |
| `best_model_*/` | Best model artifacts | Directory |

## 📊 Expected Metrics

The notebook evaluates each model on:
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction reliability
- **Recall**: Ability to find all positive cases
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Precision-recall trade-off

## 🔧 Configuration

### Modify Spark Resources
In Section 2, adjust memory allocation:
```python
.config("spark.driver.memory", "4g") \
.config("spark.executor.memory", "4g") \
```

### Change Data Source
In Section 3, modify HDFS path:
```python
hdfs_path = "hdfs://localhost:9000/your/custom/path.csv"
```

### Adjust Model Parameters
In Sections 6-8, tune hyperparameters:
```python
# Logistic Regression
lr = LogisticRegression(maxIter=100, regParam=0.01)

# Random Forest
rf = RandomForestClassifier(numTrees=100, maxDepth=10)

# GBT
gbt = GBTClassifier(maxIter=50, maxDepth=5)
```

## 📝 Cell-by-Cell Breakdown

| Cell | Section | Lines | Purpose |
|------|---------|-------|---------|
| 1 | Header | - | Notebook introduction |
| 2 | Imports | 23 | Library imports and setup |
| 3 | Spark Session | 19 | Initialize PySpark |
| 4 | Load Data | 27 | Read from HDFS |
| 5 | Preprocessing | 60 | Feature engineering |
| 6 | Split Data | 14 | Train-test split |
| 7 | Logistic Reg | 41 | Train LR model |
| 8 | Random Forest | 41 | Train RF model |
| 9 | GBT | 41 | Train GBT model |
| 10 | Evaluation | 51 | Calculate metrics |
| 11 | Confusion Matrix | 37 | Plot confusion matrices |
| 12 | ROC Curves | 53 | Plot ROC curves |
| 13 | Metrics Charts | 33 | Compare metrics |
| 14 | Feature Importance | 41 | Analyze features |
| 15 | Summary | 26 | Save best model |
| 16 | Cleanup | 6 | Stop Spark |

## ⚡ Performance Tips

1. **Large Datasets**: Increase Spark partitions
   ```python
   .config("spark.sql.shuffle.partitions", "400")
   ```

2. **Memory Issues**: Reduce data sampling
   ```python
   df_clean = df_clean.sample(fraction=0.1, seed=42)
   ```

3. **Faster Training**: Reduce model complexity
   ```python
   rf = RandomForestClassifier(numTrees=50, maxDepth=5)
   ```

## 🆚 Comparison with Other Notebooks

| Feature | sparknote.ipynb | multi_model_*.ipynb |
|---------|----------------|---------------------|
| Framework | PySpark MLlib | Scikit-learn/PyTorch |
| Data Source | HDFS | Local CSV |
| Scale | Distributed | Single machine |
| GPU | No | Yes (CUDA) |
| Models | 3 (LR, RF, GBT) | 3 (RF, XGB, NN) |

## ❓ Troubleshooting

### HDFS Connection Failed
```bash
# Check HDFS status
hdfs dfsadmin -report

# Restart if needed
stop-dfs.sh && start-dfs.sh
```

### Dataset Not Found
```bash
# Run preprocessing first
sbt "runMain CombineDatasets"
```

### Out of Memory
```python
# Increase memory in Section 2
.config("spark.driver.memory", "8g")
```

## 📚 Documentation

For detailed information, see:
- **SPARKNOTE_README.md** - Complete documentation
- **README.md** - Repository overview
- **TECHNICAL_DETAILS.md** - Architecture details

## 🎓 Learning Resources

1. [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
2. [Machine Learning Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)
3. [Binary Classification](https://spark.apache.org/docs/latest/ml-classification-regression.html)

---

**Created**: October 2025  
**Version**: 1.0  
**Author**: Automated ML Pipeline
