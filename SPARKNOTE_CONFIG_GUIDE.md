# Configuration Guide for sparknote.ipynb

## Quick Configuration for HDFS vs Local Files

The `sparknote.ipynb` notebook is configured to use HDFS by default. If you don't have HDFS set up, you can easily modify it to use local files.

## Option 1: Use HDFS (Default)

### Prerequisites:
1. HDFS must be installed and running
2. Start HDFS: `start-dfs.sh`
3. Verify: `hdfs dfs -ls /`
4. Dataset must exist at: `hdfs://localhost:9000/output/combined_dataset_complete.csv`

### No changes needed - the notebook will work as-is!

## Option 2: Use Local Files (No HDFS Required)

### Step 1: Locate Cell 2 (Spark Session Initialization)

Find this line:
```python
.config("fs.defaultFS", "hdfs://localhost:9000") \
```

Change it to:
```python
.config("fs.defaultFS", "file:///") \
```

### Step 2: Locate Cell 3 (Load Data from HDFS)

Find this line:
```python
hdfs_path = "hdfs://localhost:9000/output/combined_dataset_complete.csv"
```

Change it to:
```python
hdfs_path = "file:///absolute/path/to/your/combined_dataset_final.csv"
# Example: "file:///home/username/data/combined_dataset_final.csv"
```

**Important:** Use absolute paths! Relative paths may not work correctly.

### Step 3: Run the notebook

That's it! The rest of the notebook will work with local files.

## Quick Find & Replace Guide

If using Jupyter Notebook:
1. Open `sparknote.ipynb` in Jupyter
2. Use Edit → Find and Replace
3. Find: `hdfs://localhost:9000`
4. Replace with: `file:///absolute/path/to/your`
5. Replace All

## Alternative: Use the Python Script

We've also provided a Python script to automate the conversion:

```bash
# Create a local-files version of the notebook
python convert_notebook_to_local.py
```

This will create `sparknote_local.ipynb` with all HDFS paths replaced.

## Troubleshooting

### Issue: "Cannot connect to HDFS"
**Solution:** Either:
- Start HDFS: `start-dfs.sh`
- OR switch to local files (see Option 2 above)

### Issue: "File not found" with local files
**Solution:** 
- Use absolute paths, not relative
- Verify file exists: `ls -l /absolute/path/to/file.csv`
- Check file permissions

### Issue: "URI scheme is not 'file'"
**Solution:** 
- Ensure you changed `fs.defaultFS` config to `file:///`
- Use `file:///` prefix in all file paths (three slashes!)

## Best Practices

1. **HDFS (Recommended for large datasets):**
   - Use when working with multi-GB datasets
   - Enables distributed processing
   - Good for production environments

2. **Local Files (Easier for testing):**
   - Use when learning or testing
   - Good for small to medium datasets (<1GB)
   - No additional setup required

## Example Configurations

### Example 1: Local Files on Linux/macOS
```python
# Cell 2
.config("fs.defaultFS", "file:///") \

# Cell 3
hdfs_path = "file:///home/username/data/combined_dataset_final.csv"
```

### Example 2: Local Files on Windows
```python
# Cell 2
.config("fs.defaultFS", "file:///") \

# Cell 3
hdfs_path = "file:///C:/Users/username/data/combined_dataset_final.csv"
```

### Example 3: HDFS (Default)
```python
# Cell 2
.config("fs.defaultFS", "hdfs://localhost:9000") \

# Cell 3
hdfs_path = "hdfs://localhost:9000/output/combined_dataset_complete.csv"
```

## Performance Comparison

| Feature | HDFS | Local Files |
|---------|------|-------------|
| Setup Complexity | High | Low |
| Large Dataset Performance | Excellent | Good |
| Distributed Processing | Yes | No |
| Fault Tolerance | High | Low |
| Learning Curve | Steep | Easy |

## Recommendation

- **For learning/testing:** Use local files
- **For production/large data:** Use HDFS

## Need Help?

See `important.txt` for:
- HDFS installation guide
- Java version requirements
- PySpark setup
- Troubleshooting

## Quick Links

- [PySpark File I/O Documentation](https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html)
- [HDFS Documentation](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html)
