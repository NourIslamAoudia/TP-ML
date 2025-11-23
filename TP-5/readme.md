# Outlier Detection: IQR vs Z-Score

## ✅ Chosen Method: IQR (Interquartile Range)

### Why IQR instead of Z-Score?

| Criterion | Z-Score | IQR |
|-----------|---------|-----|
| Assumes normal distribution | ✅ Yes | ❌ No |
| Works with skewed data | ❌ Weak | ✅ Strong |
| Robust to extreme values | ❌ No | ✅ Yes |
| Works well with small datasets | ❌ Not ideal | ✅ Yes |

### 🔥 Key Takeaway

In most real datasets (including traffic/network data), values are **not perfectly normally distributed**, so:

**IQR is more appropriate and safer for detecting outliers.**

---

## What is IQR?

The **Interquartile Range (IQR)** measures the middle 50% of your data:

```
IQR = Q3 - Q1
```

Where:
- **Q1** (First Quartile): 25th percentile
- **Q3** (Third Quartile): 75th percentile

### Outlier Detection Formula

**Lower Bound**: `Q1 - 1.5 × IQR`  
**Upper Bound**: `Q3 + 1.5 × IQR`

Any data point outside these bounds is considered an outlier.

---

## Advantages of IQR

✅ **Distribution-free**: Works with any data distribution  
✅ **Robust**: Not affected by extreme values  
✅ **Simple**: Easy to understand and implement  
✅ **Reliable**: Industry standard for exploratory data analysis  

---

## When to Use Each Method

### Use IQR when:
- Data is skewed or non-normal
- You have outliers that might skew mean/std
- Working with small datasets
- You want a robust, assumption-free method

### Use Z-Score when:
- Data is approximately normally distributed
- Large sample sizes (n > 30)
- You need to specify exactly how many standard deviations away
- Working in statistical hypothesis testing contexts

---

## Implementation Example

### Python
```python
import numpy as np

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers, lower_bound, upper_bound
```

---

## Real-World Applications

🌐 **Network Traffic Analysis**: Detecting unusual bandwidth usage  
📊 **Financial Data**: Identifying anomalous transactions  
🏥 **Healthcare**: Finding unusual patient metrics  
🏭 **Manufacturing**: Quality control and defect detection  
📈 **Sales Data**: Spotting unusual sales patterns  

---