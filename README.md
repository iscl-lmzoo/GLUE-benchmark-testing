# GLUE-benchmark-testing

**Research Questions:**
1. How easy is it to replicate the GLUE benchmark scores of different models research papers?
2. How much are the benchmark scores affected by small variations such as different seeds, dataset splits etc.? Can we trust single run benchmark scores? (targetting internal validity)
3. Do benchmark scores generalize to adversarial or out-of-distribution data? (targeting external validity)
4. What other measures can we use to evaluate our models that allow us to find the best model for a specific task?

**Experiments:**
The following experiments were conducted:

1. Baseline
  Standard fine-tuning on SST-2
  Fixed seed (42)
2. Random Seed Variation
  Seeds: 1, 11, 21, 31, 42
  Measures variance across runs
3. Training Data Subsets
  50% of training data
  25% of training data
4. Alternative Train/Validation Splits
  Different random splits (90/10)
5. Adversarial Evaluation
  Evaluation on Adversarial GLUE (SST-2)

**Models used:**
1. BERT (Devlin et al., 2019)
2. DistilBERT (Sanh et al., 2019)

