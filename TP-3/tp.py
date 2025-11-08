# 0. Imports & seeds
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Prefer tf.keras if available; fall back to standalone keras to avoid import resolution errors
try:
    keras = tf.keras
    layers = tf.keras.layers
except Exception:
    import keras as _keras
    keras = _keras
    layers = _keras.layers

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Load the data
# If you have the files locally, put the local path.
# Examples of known sources (if you want to download):
# https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
# https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt

data_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"  # -> change to your local or remote path
df = pd.read_csv(data_url, header=None)

# 2. Assign the 43 column names (according to the assignment)
cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty']
df.columns = cols

# Display the first 5 rows and shape
print("Shape:", df.shape)
print(df.head())

# 3. df.info() and label distribution
print(df.info())
print("\nLabel distribution:\n", df['label'].value_counts())

# ---------- Preprocessing ----------
# Q: remove 'difficulty'
df = df.drop(columns=['difficulty'])

# X / y
X = df.drop(columns=['label'])
y = df['label'].apply(lambda s: 0 if s == 'normal' else 1)  # binary: normal=0, attack=1

# Encoding categorical variables
X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])

print("Feature count after get_dummies:", X.shape[1])  # should be ~122 if standard dataset

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (if you're using a single dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=SEED, stratify=y.values
)
print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# ---------- Model builder ----------
def build_model(input_dim, n_hidden_layers=1, n_neurons=32, learning_rate=0.001, dropout_rate=0.0):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation='relu'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_dim = X_train.shape[1]

# ---------- Define the two models parameters ----------
# Model 1: Shallow
m1_params = {
    'n_hidden_layers': 1,
    'n_neurons': 4,
    'learning_rate': 0.05,
    'dropout_rate': 0.0,
    'batch_size': 512,
    'epochs': 15
}
# Model 2: Deep
m2_params = {
    'n_hidden_layers': 3,
    'n_neurons': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'batch_size': 64,
    'epochs': 15
}

# Build models
model_shallow = build_model(input_dim, **{k:v for k,v in m1_params.items() if k in ['n_hidden_layers','n_neurons','learning_rate','dropout_rate']})
model_deep = build_model(input_dim, **{k:v for k,v in m2_params.items() if k in ['n_hidden_layers','n_neurons','learning_rate','dropout_rate']})

# Summaries
print("Shallow model summary")
model_shallow.summary()
print("\nDeep model summary")
model_deep.summary()

# ---------- Training ----------
history_shallow = model_shallow.fit(X_train, y_train,
                                   validation_split=0.2,
                                   epochs=m1_params['epochs'],
                                   batch_size=m1_params['batch_size'],
                                   verbose=1)

history_deep = model_deep.fit(X_train, y_train,
                             validation_split=0.2,
                             epochs=m2_params['epochs'],
                             batch_size=m2_params['batch_size'],
                             verbose=1)

# Eval
test_loss_sh, test_acc_sh = model_shallow.evaluate(X_test, y_test, verbose=0)
test_loss_de, test_acc_de = model_deep.evaluate(X_test, y_test, verbose=0)
print(f"Shallow test acc: {test_acc_sh:.4f} | Deep test acc: {test_acc_de:.4f} | Diff: {test_acc_de - test_acc_sh:.4f}")

# ---------- Visualization ----------
plt.figure(figsize=(12,5))
# Training accuracy
plt.subplot(1,2,1)
plt.plot(history_shallow.history['accuracy'], label='Shallow train acc')
plt.plot(history_deep.history['accuracy'], label='Deep train acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')

# Validation accuracy
plt.subplot(1,2,2)
plt.plot(history_shallow.history['val_accuracy'], label='Shallow val acc')
plt.plot(history_deep.history['val_accuracy'], label='Deep val acc')
plt.xlabel('Epoch')
plt.ylabel('Val Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()

# Bar chart test accuracies
plt.figure(figsize=(6,4))
accs = [test_acc_sh, test_acc_de]
names = ['Shallow', 'Deep']
bars = plt.bar(names, accs)
for bar,acc in zip(bars,accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.4f}", ha='center', va='bottom')
plt.title("Test accuracy comparison")
plt.ylim(0,1)
plt.show()