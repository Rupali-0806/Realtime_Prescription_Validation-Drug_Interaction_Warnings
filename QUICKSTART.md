# Quick Start Guide - Enhanced Drug Interaction Prediction System

## Overview
This system predicts drug interaction safety for combinations of 2-10 drugs using machine learning models with CUDA acceleration.

## Key Capabilities

### 1. Check Single Drug Combination
```python
# Example: Check if Aspirin + Warfarin is safe
predictor = EnhancedDrugCombinationPredictor(saved_model_package, preprocessor, cuda_combination_kernel)

result = predictor.predict_single_combination(
    drugs=['Aspirin', 'Warfarin'],
    dosage=100.0  # Optional
)

print(f"Prediction: {result['prediction']}")           # 'safe' or 'unsafe'
print(f"Confidence: {result['confidence']:.1%}")       # e.g., 95.3%
print(f"Safe probability: {result['safe_probability']:.1%}")
```

### 2. Check ALL Drug Combinations (PARALLEL)
```python
# Example: Patient takes 5 drugs - check ALL combinations
drugs = ['Aspirin', 'Warfarin', 'Ibuprofen', 'Naproxen', 'Clopidogrel']

results = predictor.predict_all_combinations(drugs, dosage=150.0)

# Summary
print(f"Total combinations checked: {results['total_combinations']}")  # 26
print(f"Safe: {results['summary']['safe_combinations']}")
print(f"Unsafe: {results['summary']['unsafe_combinations']}")

# Detailed results
for r in results['results']:
    print(f"{' + '.join(r['drugs'])}: {r['prediction']} ({r['confidence']:.1%})")
```

### 3. Learn from New Data
```python
# After model is deployed, learn from new observations
learner = IncrementalLearner(predictor.model, preprocessor, device='cuda')

new_combinations = [
    ['NewDrug1', 'Aspirin'],
    ['NewDrug2', 'Warfarin']
]
labels = [1, 0]  # 1=unsafe, 0=safe
dosages = [100.0, 150.0]

learner.learn_from_new_data(new_combinations, labels, dosages, epochs=5)
learner.save_updated_model('updated_model.pth')
```

## Input Specifications

### Drug Inputs
- **Minimum:** 2 drugs
- **Maximum:** 10 drugs
- **Format:** List of drug names as strings
- **Example:** `['Aspirin', 'Warfarin', 'Ibuprofen']`

### Dosage (Optional)
- **Type:** Float
- **Unit:** Doses per 24 hours
- **Example:** `150.0`
- **Note:** Model works with or without dosage

### Labels (For Training/Learning)
- **Format:** Integer
- **Values:** `0` = safe, `1` = unsafe

## Output Specifications

### Single Combination Result
```python
{
    'drugs': ['Aspirin', 'Warfarin'],
    'dosage': '100.0 per 24hrs',
    'prediction': 'unsafe',
    'confidence': 0.953,
    'safe_probability': 0.047,
    'unsafe_probability': 0.953,
    'model_used': 'PyTorch Neural Network'
}
```

### All Combinations Result
```python
{
    'total_combinations': 26,
    'results': [
        {
            'drugs': ['Aspirin', 'Warfarin'],
            'prediction': 'unsafe',
            'safe_prob': 0.047,
            'unsafe_prob': 0.953,
            'confidence': 0.953
        },
        # ... more results
    ],
    'summary': {
        'safe_combinations': 20,
        'unsafe_combinations': 6,
        'safety_percentage': 76.9
    }
}
```

## Performance Guide

### Combination Counts by Drug Number

| Drugs | 2-way | 3-way | 4-way | 5-way | 6-way | 7-way | 8-way | 9-way | 10-way | **Total** |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----------|
| 2     | 1     | -     | -     | -     | -     | -     | -     | -     | -      | **1**     |
| 3     | 3     | 1     | -     | -     | -     | -     | -     | -     | -      | **4**     |
| 4     | 6     | 4     | 1     | -     | -     | -     | -     | -     | -      | **11**    |
| 5     | 10    | 10    | 5     | 1     | -     | -     | -     | -     | -      | **26**    |
| 6     | 15    | 20    | 15    | 6     | 1     | -     | -     | -     | -      | **57**    |
| 7     | 21    | 35    | 35    | 21    | 7     | 1     | -     | -     | -      | **120**   |
| 8     | 28    | 56    | 70    | 56    | 28    | 8     | 1     | -     | -      | **247**   |
| 9     | 36    | 84    | 126   | 126   | 84    | 36    | 9     | 1     | -      | **502**   |
| 10    | 45    | 120   | 210   | 252   | 210   | 120   | 45    | 10    | 1      | **1,013** |

### Processing Time (Approximate)

| Drugs | Combinations | CPU Time | GPU Time | Speedup |
|-------|--------------|----------|----------|---------|
| 3     | 4            | ~40ms    | ~5ms     | 8x      |
| 5     | 26           | ~260ms   | ~15ms    | 17x     |
| 7     | 120          | ~1.2s    | ~50ms    | 24x     |
| 10    | 1,013        | ~10s     | ~200ms   | 50x     |

*Times are approximate and depend on hardware*

## Common Use Cases

### Use Case 1: Prescription Validation
```python
# Pharmacist reviews new prescription
prescription_drugs = ['Aspirin', 'Warfarin', 'Lisinopril', 'Metformin']
results = predictor.predict_all_combinations(prescription_drugs)

if results['summary']['unsafe_combinations'] > 0:
    print("⚠️ WARNING: Unsafe interactions detected!")
    for r in results['results']:
        if r['prediction'] == 'unsafe':
            print(f"  - {' + '.join(r['drugs'])}: {r['confidence']:.1%} confidence")
```

### Use Case 2: Drug Research
```python
# Researcher tests new drug with common medications
new_drug = 'ExperimentalDrug-123'
common_drugs = ['Aspirin', 'Ibuprofen', 'Acetaminophen']

for drug in common_drugs:
    result = predictor.predict_single_combination([new_drug, drug])
    print(f"{new_drug} + {drug}: {result['prediction']}")
```

### Use Case 3: Clinical Decision Support
```python
# Doctor considers adding drug to existing regimen
current_drugs = ['Warfarin', 'Lisinopril', 'Metformin']
candidate_drug = 'Ibuprofen'

# Check candidate with all current drugs
test_combinations = [current_drugs + [candidate_drug]]
test_combinations.extend([[candidate_drug, d] for d in current_drugs])

results = predictor.analyze_multiple_combinations(test_combinations)

safe_count = sum(1 for r in results if r['prediction'] == 'safe')
if safe_count == len(results):
    print(f"✓ {candidate_drug} appears safe to add")
else:
    print(f"⚠️ {candidate_drug} may have interactions")
```

### Use Case 4: Continuous Learning
```python
# Healthcare system learns from real-world outcomes
# Collected data: patients who had adverse events

adverse_events = [
    (['NewDrug1', 'Aspirin'], 1),      # Unsafe interaction observed
    (['NewDrug2', 'Warfarin'], 1),     # Unsafe interaction observed
    (['DrugA', 'DrugB'], 0)            # Safe combination observed
]

new_combos = [combo for combo, _ in adverse_events]
new_labels = [label for _, label in adverse_events]

learner = IncrementalLearner(predictor.model, preprocessor)
learner.learn_from_new_data(new_combos, new_labels, epochs=5)
learner.save_updated_model('model_v2.pth')
```

## Model Selection

The system automatically trains 3 models and selects the best:

1. **Random Forest** - Good for interpretability
2. **XGBoost** - Best for structured data
3. **PyTorch Neural Network** - Best for complex patterns

Best model is selected based on ROC-AUC score and saved automatically.

## Requirements

```bash
# Python packages
pip install torch torchvision  # PyTorch with CUDA
pip install xgboost            # XGBoost with GPU support
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
```

## Troubleshooting

### Issue: CUDA not available
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# System will automatically fall back to CPU if CUDA unavailable
# (slower but still works)
```

### Issue: Out of memory
```python
# For large combination sets, use smaller batch sizes
cuda_kernel.batch_size = 128  # Default is 256
```

### Issue: Unknown drugs
```python
# New drugs not in training data will be treated as "UNKNOWN"
# For better predictions, use incremental learning to teach the model
```

## Best Practices

1. **Always check all combinations** for prescriptions with 3+ drugs
2. **Include dosage when available** for better predictions
3. **Use incremental learning** to keep model updated with new observations
4. **Validate high-risk combinations** manually when confidence < 80%
5. **Monitor model performance** and retrain periodically
6. **Keep drug names consistent** (standardize to generic names)

## Next Steps

1. Review `ENHANCEMENTS.md` for detailed feature documentation
2. Review `TECHNICAL_DETAILS.md` for implementation details
3. Run the notebook cells to train models
4. Use the comprehensive demo (Cell 15) to see all features
5. Integrate into your application using the predictor API

## Support

For issues or questions:
1. Check the documentation files
2. Review the notebook comments
3. Examine the demo cell for usage examples
