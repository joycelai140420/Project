#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 拆分未标记数据和标记数据
num_unlabeled = 30000  # 设定未标记数据的数量
X_unlabeled = X_train[:num_unlabeled]
X_train_labeled = X_train[num_unlabeled:]
y_train_labeled = y_train[num_unlabeled:]

# 数据预处理
X_train_labeled = X_train_labeled.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_unlabeled = X_unlabeled.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train_labeled = to_categorical(y_train_labeled, 10)
y_test = to_categorical(y_test, 10)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(X_train_labeled)


# In[2]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 初步训练模型
initial_history = model.fit(datagen.flow(X_train_labeled, y_train_labeled, batch_size=32), 
                            epochs=10, 
                            validation_data=(X_test, y_test))


# In[3]:


pseudo_labels = model.predict(X_unlabeled)
pseudo_labels = np.argmax(pseudo_labels, axis=1)
pseudo_labels = to_categorical(pseudo_labels, 10)


# In[4]:


# 合并标记数据和带伪标签的未标记数据
X_combined = np.concatenate((X_train_labeled, X_unlabeled), axis=0)
y_combined = np.concatenate((y_train_labeled, pseudo_labels), axis=0)

# 数据增强
datagen_combined = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen_combined.fit(X_combined)

# 重训练模型
final_history = model.fit(datagen_combined.flow(X_combined, y_combined, batch_size=32), 
                          epochs=10, 
                          validation_data=(X_test, y_test))


# In[5]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# 展示训练过程中的损失和准确率
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(initial_history.history['loss'], label='Initial Training Loss')
plt.plot(initial_history.history['val_loss'], label='Initial Validation Loss')
plt.plot(final_history.history['loss'], label='Final Training Loss')
plt.plot(final_history.history['val_loss'], label='Final Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(initial_history.history['accuracy'], label='Initial Training Accuracy')
plt.plot(initial_history.history['val_accuracy'], label='Initial Validation Accuracy')
plt.plot(final_history.history['accuracy'], label='Final Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Final Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

# 可视化一些测试样本的预测结果
predictions = model.predict(X_test[:10])

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {tf.argmax(y_test[i])}, Pred: {tf.argmax(predictions[i])}")
    plt.axis('off')
plt.show()


# In[ ]:




