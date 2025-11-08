import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("TP2: K-Nearest Neighbors (KNN) Algorithm in Cybersecurity")
print("="*60)

# ============================================================
# PART 1: Manual KNN for 2D Security Classification
# ============================================================
print("\n" + "="*60)
print("PART 1: Manual KNN for 2D Network Traffic Classification")
print("="*60)

# Training data
X_train_2d = np.array([[1.0, 0.25],
                       [0.4, 0.10],
                       [0.5, 0.50],
                       [1.0, 1.0]])
y_train_2d = np.array([0, 0, 1, 1])  # 0=Normal, 1=Attack

# New point to classify
P = np.array([0.5, 0.15])

# Q1: Manual calculation - Euclidean distance
print("\nQ1: Manual Euclidean Distance Calculation")
print(f"New point to classify: ({P[0]}, {P[1]})")
print("\nCalculating distances:")

distances_2d = np.sqrt(np.sum((X_train_2d - P)**2, axis=1))
for i, (point, label, dist) in enumerate(zip(X_train_2d, y_train_2d, distances_2d)):
    label_str = "Normal" if label == 0 else "Attack"
    print(f"Point {i+1}: {point} | Label: {label_str} | Distance: {dist:.4f}")

# Sort by distance
sorted_indices = np.argsort(distances_2d)
print(f"\nSorted distances: {distances_2d[sorted_indices]}")
print(f"\nFor K=3, the 3 nearest neighbors are:")
for i in range(3):
    idx = sorted_indices[i]
    label_str = "Normal" if y_train_2d[idx] == 0 else "Attack"
    print(f"  - Point {idx+1}: {X_train_2d[idx]} | Label: {label_str} | Distance: {distances_2d[idx]:.4f}")

# Majority vote for K=3
nearest_labels = y_train_2d[sorted_indices[:3]]
prediction = np.bincount(nearest_labels).argmax()
print(f"\nMajority vote (K=3): {nearest_labels}")
print(f"Predicted class: {'Normal' if prediction == 0 else 'Attack'}")

# ============================================================
# PART 2: Manual KNN for 1D Classification (Login Attempt)
# ============================================================
print("\n" + "="*60)
print("PART 2: Manual KNN for 1D Login Attempt Classification")
print("="*60)

# Training data
X_train_1d = np.array([[0], [3], [4], [6], [9]])
y_train_1d = np.array([0, 1, 1, 0, 0])  # 0=Legitimate, 1=Suspicious

# New login attempt
X_test_1d = np.array([[5.5]])
y_true_1d = np.array([1])  # True class is suspicious

# Q3: Manhattan Distance Calculation
print("\nQ3: Manhattan Distance Calculation")
print(f"New login time: {X_test_1d[0][0]}")
print("\nCalculating Manhattan distances:")

distances_1d = np.abs(X_train_1d.flatten() - X_test_1d[0][0])
data_1d = pd.DataFrame({
    'Login Time': X_train_1d.flatten(),
    'Label': y_train_1d,
    'Distance': distances_1d
})
data_1d = data_1d.sort_values('Distance')
print(data_1d.to_string(index=False))

# Q4: Prediction and Error Evaluation
print("\nQ4: Prediction and Error Evaluation")
print("True label = 1 (Suspicious)\n")

results = []
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    model.fit(X_train_1d, y_train_1d)
    y_pred = model.predict(X_test_1d)
    mae = mean_absolute_error(y_true_1d, y_pred)
    results.append({'K': k, 'Predicted': y_pred[0], 'True': y_true_1d[0], 'MAE': mae})
    print(f"K = {k}, Predicted = {y_pred[0]}, MAE = {mae:.0f}")

print(f"\nBest K = 3 (lowest MAE = 0)")

# ============================================================
# PART 3: Python Implementation on Security Dataset
# ============================================================
print("\n" + "="*60)
print("PART 3: Network Intrusion Detection Dataset")
print("="*60)

# Your existing code
data = {
    'packet_size': [200, 450, 300, 700, 120, 1000, 150, 400, 800, 130],
    'connection_time': [30, 50, 25, 80, 10, 100, 15, 45, 90, 12],
    'malicious': [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # 0=Normal, 1=Attack
}
df = pd.DataFrame(data)
print("\nDataset:")
print(df)

# Prepare and scale data
x = df[['packet_size', 'connection_time']]
y = df['malicious']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\n" + "-"*60)
print("Testing with Manhattan distance metric:")
print("-"*60)
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"K={k}, Predictions: {y_pred}, MAE: {mae:.2f}")

print("\n" + "-"*60)
print("Testing with Euclidean distance metric:")
print("-"*60)
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"K={k}, Predictions: {y_pred}, MAE: {mae:.2f}")

# ============================================================
# PART 4: KNN for Email Spam Detection
# ============================================================
print("\n" + "="*60)
print("PART 4: Email Spam Detection")
print("="*60)

# Training dataset
X_train_spam = np.array([
    [0, 1],  # Email 1
    [1, 2],  # Email 2
    [2, 3],  # Email 3
    [3, 4],  # Email 4
    [5, 6],  # Email 5
    [0, 0],  # Email 6
    [4, 5]   # Email 7
])
y_train_spam = np.array([0, 0, 1, 1, 1, 0, 1])  # 0=Normal, 1=Spam

# New email to classify
X_test_spam = np.array([[2, 1]])
y_true_spam = np.array([1])  # Assume it is actually spam

print("\nTraining Data:")
spam_df = pd.DataFrame(X_train_spam, columns=['Links', 'Spam Words'])
spam_df['Label'] = ['Normal' if l == 0 else 'Spam' for l in y_train_spam]
print(spam_df)

print(f"\nNew email to classify: Links={X_test_spam[0][0]}, Spam Words={X_test_spam[0][1]}")
print(f"True label: Spam\n")

# Q1: Plot the training data
plt.figure(figsize=(10, 6))
for label in [0, 1]:
    mask = y_train_spam == label
    plt.scatter(X_train_spam[mask, 0], X_train_spam[mask, 1], 
                label='Normal' if label == 0 else 'Spam', 
                s=100, alpha=0.7)

plt.scatter(X_test_spam[0, 0], X_test_spam[0, 1], 
            color='red', marker='X', s=200, 
            label='New Email', edgecolors='black', linewidth=2)

plt.xlabel('Number of Links', fontsize=12)
plt.ylabel('Number of Spam Keywords', fontsize=12)
plt.title('Email Spam Detection - Training Data', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spam_detection_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'spam_detection_plot.png'")

# Q2: Train, Predict, Compute MAE
print("\nQ2: KNN Predictions for different K values:")
print("-"*60)
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_spam, y_train_spam)
    y_pred = knn.predict(X_test_spam)
    mae = mean_absolute_error(y_true_spam, y_pred)
    pred_label = 'Spam' if y_pred[0] == 1 else 'Normal'
    print(f"K = {k}, Predicted = {pred_label} ({y_pred[0]}), MAE = {mae:.0f}")