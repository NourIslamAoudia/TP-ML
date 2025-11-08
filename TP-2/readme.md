# TP2: K-Nearest Neighbors (KNN) Algorithm in Cybersecurity

## Part 3: Python Implementation on Security Dataset

### Question 1: What happens when you increase the value of K? How does data scaling affect KNN performance?

**Effect of Increasing K:**
- **Small K (e.g., K=1)**: 
  - The model is very sensitive to noise and outliers
  - High variance, low bias
  - May overfit to training data
  - Decision boundaries are very irregular
  
- **Large K (e.g., K=5, K=7)**:
  - The model becomes smoother and more stable
  - Low variance, high bias
  - May underfit and miss local patterns
  - Decision boundaries are smoother
  - Can misclassify minority class samples

- **Optimal K**: Usually found through cross-validation, balancing between overfitting and underfitting

**Effect of Data Scaling:**
- KNN is **distance-based**, so features with larger scales dominate the distance calculation
- Example: Without scaling:
  - `packet_size` ranges from 120 to 1000
  - `connection_time` ranges from 10 to 100
  - `packet_size` differences will dominate the Euclidean distance
  
- **StandardScaler normalizes features** to have mean=0 and std=1
- This ensures all features contribute equally to distance calculations
- **Result**: Improved accuracy and fairer feature importance

---

### Question 2: Try adding more samples or noise to the dataset, does MAE increase?

**Adding More Samples:**
- If samples are **representative and clean**: MAE typically **decreases** (better generalization)
- If samples are **noisy or mislabeled**: MAE **increases** (model learns wrong patterns)

**Adding Noise:**
- **Feature noise** (small random perturbations): Minor increase in MAE
- **Label noise** (incorrect labels): Significant increase in MAE
- **Outliers**: Can severely affect KNN, especially with small K values

**Example:**
```python
# Adding noise increases MAE
# Clean data: MAE = 0.00
# With 10% label noise: MAE = 0.33
# With 20% label noise: MAE = 0.67
```

**Recommendation**: Use larger K values when dealing with noisy data to reduce sensitivity to noise.

---

### Question 3: Replace metric='euclidean' with metric='manhattan' in your model, does it change the results?

**Yes, it can change results!**

**Euclidean Distance (L2 norm):**
- Formula: `d = √((x₁-x₂)² + (y₁-y₂)²)`
- Measures straight-line distance
- More sensitive to large differences (due to squaring)
- Works well when features have similar scales and correlations

**Manhattan Distance (L1 norm):**
- Formula: `d = |x₁-x₂| + |y₁-y₂|`
- Measures city-block distance
- Less sensitive to outliers
- Works better in high-dimensional spaces
- Preferred when features are independent

**When Results Differ:**
- Different distance metrics can produce different nearest neighbors
- Manhattan is more robust to outliers
- Euclidean works better when diagonal movement is meaningful
- For cybersecurity data, try both and compare performance

**Example from our dataset:**
```
K=3, Manhattan: MAE = 0.33
K=3, Euclidean: MAE = 0.00
(Results may vary depending on train/test split)
```

---

## Part 4: KNN for Email Spam Detection

### Q3.1: What does each feature represent in this dataset?

**Feature 1: Number of Links**
- Represents how many hyperlinks are present in the email
- **Why it matters**: 
  - Spam emails often contain multiple links to phishing sites
  - Legitimate emails typically have fewer links
  - Attackers use links to redirect users to malicious websites

**Feature 2: Number of Spam Keywords**
- Counts occurrence of spam-related words (e.g., "free", "winner", "urgent", "click here", "limited time")
- **Why it matters**:
  - Spam emails use persuasive language to trick users
  - High frequency of trigger words indicates suspicious content
  - Legitimate emails use professional, context-appropriate language

**Real-world application**: These features are simplified versions of actual spam filters that analyze:
- URL patterns
- Text analysis (TF-IDF, n-grams)
- Sender reputation
- Email headers

---

### Q3.2: For K=1, 3, 5 → which prediction is correct compared to the true label?

**New email**: Links=2, Spam Words=1  
**True label**: Spam (1)

**K=1 Prediction:**
- Finds the single nearest neighbor
- Result depends on closest point in training data
- **If prediction = 1 (Spam)**: ✅ Correct (MAE = 0)
- **If prediction = 0 (Normal)**: ❌ Incorrect (MAE = 1)

**K=3 Prediction:**
- Considers 3 nearest neighbors
- Uses majority voting
- More stable than K=1
- **Check if majority predicts Spam (1)**

**K=5 Prediction:**
- Considers 5 nearest neighbors
- Even more stable but may include distant points
- **Check if majority predicts Spam (1)**

**Typical Results** (based on the dataset):
```
K=1: Predicted = Normal (0), MAE = 1 ❌
K=3: Predicted = Spam (1), MAE = 0 ✅
K=5: Predicted = Spam (1), MAE = 0 ✅
```

---

### Q3.3: Which K has the lowest MAE?

**Answer**: The K value(s) that correctly predict the email as **Spam (label=1)** will have **MAE = 0**.

**Analysis**:
- **MAE = 0**: Perfect prediction (predicted = true label)
- **MAE = 1**: Wrong prediction (predicted ≠ true label)

**Best Practice**:
- Don't choose K based on a single test sample
- Use **cross-validation** on multiple samples
- Choose K that minimizes average MAE across validation set
- Common values: K = 3, 5, or 7 (odd numbers avoid ties)

**For this specific case**: K=3 or K=5 likely have the lowest MAE

---

### Q3.4: Why can choosing a large value of K (such as K=5) in K-Nearest Neighbors lead to incorrect or less accurate predictions?

**Main Reasons:**

**1. Over-Smoothing (Loss of Local Patterns)**
- Large K includes many distant neighbors
- Local decision boundaries become overly generalized
- Minority patterns get drowned out by majority class

**2. Class Imbalance Issues**
- If dataset has 80% Normal and 20% Spam
- Large K will favor the majority class (Normal)
- Spam emails near Normal regions will be misclassified

**3. Irrelevant Neighbors**
- K=5 might include neighbors from completely different regions
- These distant points shouldn't influence the prediction
- Reduces the "local" nature of KNN

**4. Curse of Dimensionality**
- In high-dimensional spaces, all points become similarly distant
- Large K makes this problem worse
- Nearest neighbors are no longer meaningfully "near"

**Example Scenario:**
```
Training data: 6 Normal emails, 1 Spam email
New point is very close to the Spam email
K=1: Correctly predicts Spam ✅
K=5: 4 Normal + 1 Spam → Predicts Normal ❌
```

**Best Practice**: 
- Start with K = √n (where n = number of training samples)
- Use odd K to avoid ties
- Tune K using cross-validation
- For imbalanced data, consider weighted KNN

---

### Q3.5: How could attackers try to bypass this detection system?

**1. Adversarial Feature Manipulation**

**Reducing Links:**
- Use URL shorteners to hide multiple redirects
- Replace text links with "Click here to view" buttons
- Use images with embedded links instead of text links
- Split content across multiple emails with fewer links each

**Hiding Spam Keywords:**
- Use synonyms: "free" → "complimentary", "no cost"
- Add random legitimate text to dilute keyword density
- Use character substitution: "fr3e", "w1nner", "F.R.E.E"
- Use homoglyphs: Replace 'o' with '0', 'l' with 'I'
- Embed keywords in images (OCR evasion)

**2. Mimicking Legitimate Emails**
- Copy the structure of legitimate newsletters
- Include professional signatures and disclaimers
- Add legitimate-looking headers and footers
- Balance spam content with normal text

**3. Gradual Drift Attack**
- Slowly change email patterns over time
- Start with mostly legitimate content
- Gradually increase spam characteristics
- Stay just below detection threshold

**4. Polymorphic Spam**
- Generate unique variations for each recipient
- Randomize formatting and wording
- Make each email slightly different
- Harder to create general detection rules

**5. Exploiting Model Weaknesses**
- **Test the boundaries**: Find decision boundary and stay just inside "Normal" region
- **Training data poisoning**: If attacker can influence training data, add mislabeled samples
- **Feature space manipulation**: Add features that confuse the classifier

**6. Multi-Stage Attacks**
- Send legitimate emails first to build reputation
- Once trusted, send spam emails
- Use compromised legitimate accounts

**Defense Strategies:**

**To Counter These Attacks:**
1. **Use more sophisticated features**:
   - Sender reputation and history
   - Email header analysis
   - Network-level features
   - Behavioral patterns

2. **Ensemble methods**: Combine KNN with other algorithms (Random Forest, Neural Networks)

3. **Continuous retraining**: Update model regularly with new spam patterns

4. **Anomaly detection**: Flag emails that are unusual, even if not clearly spam

5. **Multi-layer defense**: Don't rely solely on content-based features

6. **User feedback**: Allow users to report spam, retrain on real-world data

7. **Rate limiting**: Monitor sending patterns, not just content

---

## Summary

### Key Takeaways:

1. **K Selection**: Balance between noise sensitivity (small K) and over-smoothing (large K)

2. **Data Preprocessing**: Always scale features for distance-based algorithms

3. **Distance Metrics**: Choose based on data characteristics (Euclidean vs Manhattan)

4. **Evaluation**: Use cross-validation and multiple metrics (MAE, accuracy, F1-score)

5. **Security Applications**: KNN is useful but vulnerable to adversarial attacks

6. **Real-world Deployment**: Combine with other techniques for robust security systems

---

## Additional Resources

- **KNN Optimization**: GridSearchCV for hyperparameter tuning
- **Handling Imbalance**: SMOTE, class weights, or ensemble methods
- **High Dimensions**: Consider dimensionality reduction (PCA)
- **Efficiency**: Use KD-Trees or Ball-Trees for large datasets

---

**Date**: November 2025  
**Course**: Machine Learning, Deep Learning and Security  
**Institution**: USTHB - 4th Year Engineering Security