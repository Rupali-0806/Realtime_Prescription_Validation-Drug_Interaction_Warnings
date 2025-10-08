# PySpark MLlib Drug Interaction Prediction - sparknote.ipynb

## Overview
`sparknote.ipynb` is a comprehensive Jupyter notebook that implements distributed machine learning using PySpark MLlib to predict drug interaction safety. It trains and evaluates three different machine learning models on the complete dataset stored in HDFS.

## Features

### 🤖 Three Machine Learning Models
1. **Logistic Regression** - Binary classification with regularization and scaled features
2. **Random Forest Classifier** - Ensemble method with 100 trees and depth of 10
3. **Gradient Boosted Trees (GBT)** - Advanced boosting with 50 iterations

### 📊 Comprehensive Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Visualizations**:
  - Confusion matrices for all models
  - ROC curves comparison
  - Metrics comparison bar charts
  - Feature importance analysis

### 🚀 Key Capabilities
- Direct HDFS data loading from `hdfs://localhost:9000/output/combined_dataset_complete.csv`
- Complete PySpark ML pipeline with preprocessing, training, and evaluation
- Automatic best model selection and persistence
- High-quality visualizations saved as PNG files

## Prerequisites

### Software Requirements
- Apache Spark 3.x
- PySpark
- Jupyter Notebook or JupyterLab
- Python 3.8+
- HDFS (Hadoop Distributed File System)

### Python Libraries
```bash
pip install pyspark jupyter matplotlib seaborn pandas numpy
```

### HDFS Setup
Ensure HDFS is running and the dataset exists at:
```
hdfs://localhost:9000/output/combined_dataset_complete.csv
```

You can verify with:
```bash
hdfs dfs -ls hdfs://localhost:9000/output/
```

## How to Run

### 1. Start HDFS (if not already running)
```bash
start-dfs.sh
```

### 2. Verify Dataset Exists
```bash
hdfs dfs -ls hdfs://localhost:9000/output/combined_dataset_complete.csv
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook sparknote.ipynb
```

### 4. Execute All Cells
In Jupyter:
- Click "Cell" → "Run All"
- Or press `Shift + Enter` for each cell

## Notebook Structure

The notebook contains 16 cells organized into the following sections:

1. **Section 1**: Environment Setup and Imports
2. **Section 2**: Initialize Spark Session
3. **Section 3**: Load Data from HDFS
4. **Section 4**: Data Preprocessing
5. **Section 5**: Train-Test Split (80/20)
6. **Section 6**: Model 1 - Logistic Regression
7. **Section 7**: Model 2 - Random Forest Classifier
8. **Section 8**: Model 3 - Gradient Boosted Trees
9. **Section 9**: Model Evaluation and Metrics
10. **Section 10**: Confusion Matrices
11. **Section 11**: ROC Curves
12. **Section 12**: Metrics Comparison Visualization
13. **Section 13**: Feature Importance Visualization
14. **Section 14**: Final Summary and Model Persistence
15. **Section 15**: Cleanup

## Output Files

The notebook generates the following files:

1. **confusion_matrices.png** - Side-by-side confusion matrices for all three models
2. **roc_curves.png** - ROC curve comparison showing AUC scores
3. **metrics_comparison.png** - Bar charts comparing all metrics across models
4. **feature_importance.png** - Feature importance for Random Forest and GBT
5. **best_model_[model_name]/** - Directory containing the saved best model

## Dataset Features

The notebook processes the following features:
- **Drug Columns**: drug1, drug2, drug3 (indexed for ML)
- **Numerical Features**: 
  - `total_drugs`: Number of drugs in the prescription
  - `doses_per_24_hrs_numeric`: Dosage information (if available)
- **Target Label**: `safety_label` (safe/unsafe) → converted to binary (0/1)

## Model Configurations

### Logistic Regression
- Max Iterations: 100
- Regularization Parameter: 0.01
- Feature Scaling: StandardScaler applied

### Random Forest
- Number of Trees: 100
- Max Depth: 10
- Min Instances Per Node: 1

### Gradient Boosted Trees
- Max Iterations: 50
- Max Depth: 5
- Step Size: 0.1

## Performance Metrics

The notebook evaluates all models using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve

## Best Model Selection

The notebook automatically:
1. Compares all three models based on ROC-AUC score
2. Identifies the best performing model
3. Saves the best model to disk for deployment

## Comparison with multi_model_drug_interaction_prediction.ipynb

| Feature | sparknote.ipynb | multi_model_drug_interaction_prediction.ipynb |
|---------|----------------|----------------------------------------------|
| Framework | PySpark MLlib | Scikit-learn, XGBoost, PyTorch |
| Data Source | HDFS | Local CSV file |
| Scalability | Distributed, handles large datasets | Single machine |
| Models | Logistic Regression, Random Forest, GBT | Random Forest, XGBoost, PyTorch Neural Network |
| GPU Support | No | Yes (CUDA acceleration) |
| Feature Engineering | Basic (one-hot encoding, indexing) | Advanced (embeddings, custom features) |

## Troubleshooting

### Issue: Cannot connect to HDFS
**Solution**: 
```bash
# Check HDFS status
hdfs dfsadmin -report

# Restart HDFS if needed
stop-dfs.sh
start-dfs.sh
```

### Issue: Dataset not found
**Solution**:
```bash
# Verify the dataset exists
hdfs dfs -ls hdfs://localhost:9000/output/

# If missing, run the CombineDatasets.scala script first
sbt "runMain CombineDatasets"
```

### Issue: Out of memory
**Solution**: Increase Spark driver/executor memory in Section 2:
```python
.config("spark.driver.memory", "8g") \
.config("spark.executor.memory", "8g") \
```

### Issue: Spark not found
**Solution**:
```bash
# Install PySpark
pip install pyspark

# Or set SPARK_HOME environment variable
export SPARK_HOME=/path/to/spark
export PATH=$SPARK_HOME/bin:$PATH
```

## Notes

- The notebook uses a 80/20 train-test split with stratification
- All models are evaluated on the same test set for fair comparison
- Feature engineering focuses on the first 3 drug columns to keep the feature space manageable
- The notebook includes comprehensive logging and progress indicators
- All visualizations are saved automatically with high resolution (300 DPI)

## Next Steps

After running the notebook:
1. Review the generated visualizations to understand model performance
2. Check the best model directory for deployment artifacts
3. Use the saved model in production applications
4. Consider hyperparameter tuning for further optimization

## Support

For issues or questions:
- Check the HDFS logs: `$HADOOP_HOME/logs/`
- Check Spark logs: Available in the Jupyter notebook output
- Verify all prerequisites are installed correctly
