# 🧪 Quick Manual Test Guide for Zaura Health

## Test These Drug Combinations in Your Browser

### 🟢 SAFE Combinations (Should show GREEN)

**2-Drug Tests:**
1. `aspirin` + `metformin` (dosages: 81mg, 500mg)
2. `lisinopril` + `amlodipine` (dosages: 10mg, 5mg)
3. `acetaminophen` + `ibuprofen` (dosages: 500mg, 200mg)

**3-Drug Tests:**
4. `lisinopril` + `amlodipine` + `hydrochlorothiazide` (dosages: 10mg, 5mg, 12.5mg)
5. `aspirin` + `clopidogrel` + `atorvastatin` (dosages: 81mg, 75mg, 20mg)

**4-Drug Test:**
6. `lisinopril` + `metformin` + `atorvastatin` + `aspirin` (dosages: 10mg, 500mg, 20mg, 81mg)

---

### 🔴 UNSAFE Combinations (Should show RED)

**2-Drug Tests:**
7. `warfarin` + `aspirin` (dosages: 5mg, 325mg) - *Bleeding risk*
8. `sildenafil` + `nitroglycerin` (dosages: 100mg, 0.4mg) - *Hypotension*
9. `lithium` + `furosemide` (dosages: 600mg, 80mg) - *Toxicity*

**3-Drug Tests:**
10. `warfarin` + `aspirin` + `clopidogrel` (dosages: 5mg, 325mg, 75mg) - *Extreme bleeding*
11. `tramadol` + `sertraline` + `sumatriptan` (dosages: 100mg, 100mg, 100mg) - *Serotonin syndrome*

**4-Drug Test:**
12. `tramadol` + `fluoxetine` + `linezolid` + `dextromethorphan` (dosages: 100mg, 40mg, 600mg, 30mg) - *Severe serotonin syndrome*

---

## 🔑 Login Information
- **Doctor Access**: Username: `dr_smith`, Password: `doctor123`
- **Scientist Access**: Username: `scientist_jane`, Password: `science123`

---

## 🎯 What to Look For

### Expected Behavior:
- ✅ **Safe combinations**: Green indicators, low risk scores
- ❌ **Unsafe combinations**: Red indicators, high risk scores
- 📊 **Dosage sensitivity**: Same drugs should be safer at lower doses
- 🔢 **Multi-drug complexity**: More drugs should generally increase risk

### Model Validation:
- The AI should be **conservative** (better to warn unnecessarily than miss danger)
- **Dosage matters**: 81mg aspirin is much safer than 325mg with other drugs
- **Clinical relevance**: Real doctors use these combinations daily

---

## 📝 Testing Steps

1. **Start the app**: Run `python enhanced_app.py` in the Zaura Health directory
2. **Open browser**: Go to `http://localhost:5000`
3. **Login**: Use doctor credentials above
4. **Test combinations**: Enter drugs and dosages from the lists
5. **Check results**: Compare AI predictions with expected outcomes
6. **Note discrepancies**: Any unexpected results for model improvement

---

## 🏥 Clinical Context

### Why These Matter:
- **Aspirin + Warfarin**: Classic bleeding risk combination
- **Serotonin drugs**: Can cause life-threatening serotonin syndrome  
- **Cardiac medications**: Interactions can cause dangerous heart rhythms
- **Diabetes drugs**: Wrong combinations cause severe low blood sugar

### Real-World Usage:
- Doctors see these combinations daily
- The AI helps catch dangerous interactions
- Conservative predictions protect patients
- Dosage awareness is crucial for safety

---

**Remember**: This AI is designed to be cautiously conservative for patient safety! 🏥✨