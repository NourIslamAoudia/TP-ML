import pandas as pd


data={
    'packet_size': [200,450,300,700,120,1000,150,400,800,130],
    'connection_time': [30,50,25,80,10,100,15,45,90,12],
    'malicious': [0,0,0,1,0,1,0,0,1,0]
}

df=pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x=df[['packet_size','connection_time']]
y=df['malicious']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



print("------------------------------------------------------")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
for k in [1,3,5]:
    model=KNeighborsClassifier(n_neighbors=k, metric='manhattan')# and test with metic='euclidean'
    model.fit(x_train_scaled,y_train)
    y_pred=model.predict(x_test_scaled)
    mae= mean_absolute_error(y_test,y_pred)
    print(f"k={k}, Predictions: {y_pred}, MAE: {mae:.2f}")