# GLUE-benchmark-testing

## Research Questions

1. **Replicability:**  
   How easy is it to replicate the GLUE benchmark scores reported in research papers?

2. **Internal Validity:**  
   How much are benchmark scores affected by small variations such as:
   - different random seeds  
   - dataset splits  
   
   Can we trust single-run benchmark scores?

3. **External Validity:**  
   Do benchmark scores generalize to adversarial or out-of-distribution data?

4. **Evaluation Beyond Benchmarks:**  
   What additional evaluation measures can help identify the best model for a specific task?

---

## Experiments

The following experiments were conducted on the SST-2 task:

### 1. Baseline
- Standard fine-tuning on SST-2  
- Fixed random seed: `42`

### 2. Random Seed Variation
- Seeds: `1, 11, 21, 31, 42`  
- Measures variance across runs  

### 3. Training Data Subsets
- 50% of training data  
- 25% of training data  

### 4. Alternative Train/Validation Splits
- Different random splits (90/10)

### 5. Adversarial Evaluation
- Evaluation on **Adversarial GLUE (SST-2)**  

---

## Models Used

- **BERT** (Devlin et al., 2019)  
- **DistilBERT** (Sanh et al., 2019)  
