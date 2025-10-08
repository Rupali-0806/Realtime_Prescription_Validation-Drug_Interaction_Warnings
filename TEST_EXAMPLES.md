# Zaura Health - Drug Interaction Test Examples
# Comprehensive test cases for 2, 3, and 4 drug combinations
# Mix of safe and unsafe interactions for thorough testing

## 2-Drug Combinations

### SAFE Combinations (2 drugs)
1. **Aspirin + Metformin**
   - Aspirin: 81mg (low-dose)
   - Metformin: 500mg
   - Expected: SAFE (common diabetes + cardiovascular combination)

2. **Lisinopril + Amlodipine**
   - Lisinopril: 10mg
   - Amlodipine: 5mg
   - Expected: SAFE (common hypertension combination)

3. **Omeprazole + Levothyroxine**
   - Omeprazole: 20mg
   - Levothyroxine: 100mcg
   - Expected: SAFE (with proper timing - take separately)

4. **Simvastatin + Ezetimibe**
   - Simvastatin: 20mg
   - Ezetimibe: 10mg
   - Expected: SAFE (cholesterol management combination)

5. **Acetaminophen + Ibuprofen**
   - Acetaminophen: 500mg
   - Ibuprofen: 200mg
   - Expected: SAFE (alternating pain relief)

### UNSAFE Combinations (2 drugs)
6. **Warfarin + Aspirin**
   - Warfarin: 5mg
   - Aspirin: 325mg
   - Expected: UNSAFE (bleeding risk - double anticoagulation)

7. **Metformin + Glyburide**
   - Metformin: 1000mg
   - Glyburide: 10mg
   - Expected: UNSAFE (severe hypoglycemia risk with high doses)

8. **Sildenafil + Nitroglycerin**
   - Sildenafil: 100mg
   - Nitroglycerin: 0.4mg
   - Expected: UNSAFE (dangerous hypotension)

9. **Lithium + Furosemide**
   - Lithium: 600mg
   - Furosemide: 80mg
   - Expected: UNSAFE (lithium toxicity risk)

10. **Digoxin + Amiodarone**
    - Digoxin: 0.25mg
    - Amiodarone: 200mg
    - Expected: UNSAFE (digoxin toxicity)

## 3-Drug Combinations

### SAFE Combinations (3 drugs)
11. **Lisinopril + Amlodipine + Hydrochlorothiazide**
    - Lisinopril: 10mg
    - Amlodipine: 5mg
    - Hydrochlorothiazide: 12.5mg
    - Expected: SAFE (triple therapy for hypertension)

12. **Metformin + Glipizide + Pioglitazone**
    - Metformin: 500mg
    - Glipizide: 5mg
    - Pioglitazone: 15mg
    - Expected: SAFE (diabetes triple therapy at moderate doses)

13. **Aspirin + Clopidogrel + Atorvastatin**
    - Aspirin: 81mg
    - Clopidogrel: 75mg
    - Atorvastatin: 20mg
    - Expected: SAFE (post-cardiac event therapy)

### UNSAFE Combinations (3 drugs)
14. **Warfarin + Aspirin + Clopidogrel**
    - Warfarin: 5mg
    - Aspirin: 325mg
    - Clopidogrel: 75mg
    - Expected: UNSAFE (triple anticoagulation - extreme bleeding risk)

15. **Tramadol + Sertraline + Sumatriptan**
    - Tramadol: 100mg
    - Sertraline: 100mg
    - Sumatriptan: 100mg
    - Expected: UNSAFE (serotonin syndrome risk)

16. **Digoxin + Amiodarone + Furosemide**
    - Digoxin: 0.25mg
    - Amiodarone: 200mg
    - Furosemide: 40mg
    - Expected: UNSAFE (multiple toxicity risks)

## 4-Drug Combinations

### SAFE Combination (4 drugs)
17. **Lisinopril + Metformin + Atorvastatin + Aspirin**
    - Lisinopril: 10mg
    - Metformin: 500mg
    - Atorvastatin: 20mg
    - Aspirin: 81mg
    - Expected: SAFE (comprehensive cardiovascular + diabetes management)

18. **Omeprazole + Clarithromycin + Amoxicillin + Bismuth**
    - Omeprazole: 20mg
    - Clarithromycin: 500mg
    - Amoxicillin: 1000mg
    - Bismuth: 525mg
    - Expected: SAFE (H. pylori eradication therapy)

### UNSAFE Combinations (4 drugs)
19. **Warfarin + Aspirin + Phenytoin + Rifampin**
    - Warfarin: 5mg
    - Aspirin: 325mg
    - Phenytoin: 300mg
    - Rifampin: 600mg
    - Expected: UNSAFE (multiple drug interactions affecting warfarin)

20. **Tramadol + Fluoxetine + Linezolid + Dextromethorphan**
    - Tramadol: 100mg
    - Fluoxetine: 40mg
    - Linezolid: 600mg
    - Dextromethorphan: 30mg
    - Expected: UNSAFE (severe serotonin syndrome risk)

## Testing Instructions

### How to Test in Zaura Health:

1. **Login Options:**
   - Doctor: `dr_smith` / `doctor123`
   - Scientist: `scientist_jane` / `science123`

2. **Test Process:**
   - Enter each drug combination above
   - Include the specified dosages
   - Compare the AI prediction with the expected result
   - Note any discrepancies for model improvement

3. **Expected Behavior:**
   - Safe combinations should show GREEN indicators
   - Unsafe combinations should show RED indicators
   - Dosage-dependent analysis should consider the amounts
   - Multi-drug analysis should evaluate all interactions

### Key Learning Points:

**Safe Patterns:**
- Low-dose aspirin with most medications
- Standard diabetes/hypertension combinations
- Properly dosed multi-therapy regimens

**Unsafe Patterns:**
- Multiple anticoagulants (bleeding risk)
- High-dose combinations of similar drugs
- Serotonin syndrome combinations
- Drug interactions affecting metabolism

**Dosage Dependencies:**
- Low-dose aspirin (81mg) vs high-dose (325mg)
- Moderate vs high doses of diabetes medications
- Standard therapeutic vs excessive doses

### Model Validation Notes:

The AI model should demonstrate:
1. **Conservative healthcare approach** - When uncertain, predict unsafe
2. **Dosage awareness** - Same drugs, different safety at different doses
3. **Multi-drug complexity** - More drugs = higher interaction probability
4. **Clinical relevance** - Real-world prescription patterns

Use these examples to thoroughly test all features of the Zaura Health system and validate the model's clinical appropriateness.