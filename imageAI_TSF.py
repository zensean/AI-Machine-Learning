import tensorflow as tf
print(tf.__version__)
# 載入 fashion_mnist 數據集
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()


training_images = training_images/255.0
test_images = test_images/255.0
# 定義神經網路模型
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# 設定優化函數及損失函數
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')
# 訓練模型
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
# 進行圖像分類及預測 test 值為9
print(classifications[0])
print(test_labels[0])