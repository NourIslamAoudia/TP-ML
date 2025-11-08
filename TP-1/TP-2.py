import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.neighbors import KNeighborsClassifier
from  sklearn.metrics import mean_absolute_error

x_train = np.array([[1.0,0.25],
                     [0.4,0.10],
                     [0.5,0.50],
                     [1.0,1.0]])
y_train = np.array([0,0,1,1])

p=np.array([0.5,0.15])
distances =np.sqrt(np.sum((x_train - p)**2,axis=1))
print("distances:",distances)

x_train = np.array([[0],[3],[4],[6],[9]]),
y_train = np.array([0,1,1,0,0])
x_test =np.array([[5.5]])
y_true= np.array([1])

for k in [1,3,5]:
    model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mae= mean_absolute_error(y_true,y_pred)
    print(f"k={k}, Prediction: {y_pred}, MAE: {mae}")