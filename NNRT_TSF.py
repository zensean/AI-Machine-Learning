# neural_network_regression_training
# 神經網路迴歸訓練
import tensorflow as tf
import numpy as np
from tensorflow import keras
# 建立類神經網路。有一個具有一個神經元的層，輸入的圖形是一個值。
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# 建立優化函數及損失函數
model.compile(optimizer='sgd', loss='mean_squared_error')
# 提供資料
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
# 開始訓練類神經網路
model.fit(xs, ys, epochs=100)
# 使用模型進行預測,結果應該非常接近31
print(model.predict([10.0]))