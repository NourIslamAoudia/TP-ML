# Neural Network Intrusion Detection System - NSL-KDD Dataset

## Project Overview
This project implements and compares two neural network architectures (shallow vs. deep) for binary classification of network intrusion detection using the NSL-KDD dataset.

---

## Dataset Information

### NSL-KDD Dataset
- **Source:** https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
- **Test Set:** https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
- **Sample Size:** ~125,973 training samples
- **Features:** 41 original features (122 after one-hot encoding)
- **Target:** Binary classification (normal vs. attack)

---

## Part 2 — Initial Questions

### Q1: Why set seeds?
**For reproducibility:** Random initializations (weights, shuffle, split) produce the same results when seeds are fixed (random, numpy, tf). Reproducibility allows for fair experiment comparison.

### Q2: Number of samples?
Depends on the file you load. Display `df.shape` to see the total number of rows.  
**Example:** KDDTrain+ ~ 125,973 rows

### Q3: Number of features (excluding label & difficulty)?
- **Before categorical encoding:** 41 features
- **Total columns in file:** 43 (41 features + label + difficulty)
- **After removing difficulty:** 42 columns
- **After one-hot encoding:** ~122 features (expected after `get_dummies` applied to categorical columns)

### Q4: Distribution of normal vs. attack?
Check `df['label'].value_counts()`. NSL-KDD is often imbalanced (many more DoS attacks or conversely many more normal samples depending on the split). This distribution is important.

### Q5: Why is this important?
An imbalanced dataset can lead the model to predict the majority class (misleading accuracy). Additional metrics should be considered (precision/recall/F1) and possibly resample or use class weights.

---

## Part 3 — Preprocessing

### Q6: Encoding used for categorical variables?
**`pd.get_dummies()` (one-hot encoding)**  
**Why:** No inherent order between categories, required for neural networks.

### Q7: How many features after encoding? Why the increase?
**~122 features**  
Increases because each category of a categorical variable becomes a binary column.

### Q8: Type of classification?
**Binary classification** (normal=0, attack=1), even though the original label contains multiple attack types: here we group everything into 1.

### Q9: Why feature scaling?
Neural networks use optimizers that converge better if features have mean close to 0 and similar variance. Avoids activation saturation, accelerates learning.

### Q10: Train/test size?
With `test_size=0.2`:
- **Train:** 0.8 * N samples
- **Test:** 0.2 * N samples

---

## Part 4 — Model Architecture

### Q11: If shallow network has 4 neurons, what's the risk?
**Bottleneck:** Massive information loss (122 → 4). Underfitting, inability to model complexity, low learning capacity.

### Q12: Why sigmoid activation at output?
For binary classification it gives a probability between 0 and 1; combined with `binary_crossentropy` loss.

### Q13: Role of dropout?
**Regularization:** Reduces overfitting by randomly deactivating neurons during training, forces the network to learn more robust representations.

### Q14: Number of parameters

#### Shallow Model (122 inputs → Dense(4) → Dense(1))
- **Dense(4):** 122×4 weights + 4 bias = 488 + 4 = 492
- **Dense(1):** 4×1 + 1 = 5
- **Total = 497 trainable parameters**
- **Compression ratio:** 122 / 4 = 30.5 (reduce input space ~30x)

#### Deep Model (122 → 32 → 32 → 32 → 1)
- **Layer1:** 122×32 + 32 = 3,936
- **Layer2:** 32×32 + 32 = 1,056
- **Layer3:** 32×32 + 32 = 1,056
- **Output:** 32×1 + 1 = 33
- **Total = 6,081 trainable parameters**

### Q15: Which model will perform better? Why?
**The deep model:** Greater learning capacity, able to capture non-linear and hierarchical relationships. Shallow risks being too limited.

---

## Part 5 — Training

### Q16: Meaning of validation_split=0.2?
During training, 20% of training data (X_train) is used for validation (not used for weight updates). Serves to monitor generalization.

### Q17: Stability observations between the two models?
- **Shallow:** Often unstable, oscillations, underfitting
- **Deep:** More stable training if it has an appropriate learning rate and regularization

---

## Part 6 — Results Analysis

### Q18/Q19: Performance Comparison
Empirical results (after training) show that the **deep model has better test accuracy**. The difference depends on hyperparameters. This corresponds to the Q15 prediction.

### Q20: Why does shallow underperform?
- **Bottleneck** (too few neurons)
- **High learning rate (0.05)** can cause overshooting
- **Large batch size (512)** smooths gradients and prevents converging on fine minima

### Q21: Why does deep work better?
- More neurons + layers = **better capacity**
- Smaller lr (0.001) = **finer learning**
- Smaller batch size (64) = noisier gradients but **favors generalization**
- Dropout = **regularization**

### Q22: Training Duration
Even with same epochs, deep has more capacity and converges to richer representations; shallow lacks capacity and plateaus quickly.

---

## Part 7 — Visualization (Q23-Q25)

### Q23: Expected Patterns
- **Deep:** Ascending accuracy curves with validation close to training (if well regulated)
- **Shallow:** Low training accuracy and plateau

### Q24: Evidence of Overfitting
If training accuracy >> validation accuracy. Deep can overfit if dropout absent or epochs too high.

### Q25: For Non-Technical Audience
Show a simple chart (bars + text):  
*"The deep model detects attacks better: +X% effectiveness on the test set."*  
**Explain:** More internal "brains" = better learning.

---

## Part 8 — Experiments (Q26-Q28)

### Experiment 1 (Worse Performance)
**Change:** Reduce neurons 4→2  
**Result:** Accuracy decreases further (less capacity)  
**Why:** Bottleneck reinforced, more information loss

### Experiment 2 (Improves Shallow)
**Change:** Increase neurons → 64, lr→0.001, batch→64  
**Result:** Improved shallow can approach or even compete with deep  
**Interpretation:** Shows that network capacity and hyperparameter tuning are crucial. A single layer with many neurons can learn a lot but lacks hierarchical abstraction; however, if sufficiently wide and well optimized it can work correctly.

---

## Model Hyperparameters

### Shallow Model
```python
n_hidden_layers: 1
n_neurons: 4
learning_rate: 0.05
dropout_rate: 0.0
batch_size: 512
epochs: 15
```

### Deep Model
```python
n_hidden_layers: 3
n_neurons: 32
learning_rate: 0.001
dropout_rate: 0.2
batch_size: 64
epochs: 15
```

---

## Installation & Usage

### Requirements
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Running the Code
1. Download the NSL-KDD dataset
2. Update the `data_url` variable with your local path
3. Run the Python script

```python
python intrusion_detection.py
```

---

## Practical Tips / Debugging

### Data Loading Issues
- If `pd.read_csv` fails due to different separators, add `sep=','` or `sep='\t'`
- Check labels: sometimes the label column contains `normal.` (with period)  
  Adapt the condition using `s.startswith('normal')` or `s == 'normal.'`

### Performance Optimization
- If GPU available, install the GPU version of TensorFlow to accelerate training
- For class imbalance: use `class_weight` in `model.fit` or metrics based on precision/recall

---

## Project Structure
```
├── intrusion_detection.py    # Main implementation
├── README.md                  # This file
└── KDDTrain+.txt             # Dataset (to be downloaded)
```

---

## Results Visualization

The code generates:
1. **Training Accuracy Curves** - Compare shallow vs. deep training progress
2. **Validation Accuracy Curves** - Monitor generalization
3. **Test Accuracy Bar Chart** - Final performance comparison

---

## Future Improvements

1. Implement multi-class classification (detect specific attack types)
2. Add confusion matrix and classification report
3. Experiment with different architectures (CNN, LSTM)
4. Hyperparameter tuning with grid search
5. Handle class imbalance with SMOTE or class weights

---

## License
MIT License

## References
- NSL-KDD Dataset: https://github.com/defcom17/NSL_KDD
- Original KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

---

## Contact
For questions or improvements, please open an issue or submit a pull request.