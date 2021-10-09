import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')
# 데이터에 널값의 개수
# print(data.isnull().sum())
# 데이터에 있는 null값의 행을 제거
data = data.dropna()
# print(data.isnull().sum())
# 데이터에 있는 null값을 '100'으로 채워줌
# data = data.fillna(100)

y_data = data['admit'].values # csv에 있는 데이터를 리스트로
x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_data), np.array(y_data), epochs=1000)

# 예측
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)