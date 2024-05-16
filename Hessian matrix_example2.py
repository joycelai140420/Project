#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将标签进行one-hot编码
y = tf.keras.utils.to_categorical(y, 3)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=4, activation='relu', dtype='float64'),
    tf.keras.layers.Dense(10, activation='relu', dtype='float64'),
    tf.keras.layers.Dense(3, activation='softmax', dtype='float64')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[9]:


history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)


# In[10]:


'''
要确保有些梯度可能为None，导致hessian_matrix_flat为空。
我们需要确保在计算特征值时，只使用有效的Hessian矩阵元素。

'''
def get_hessian(model, X):
    with tf.GradientTape() as tape:
        loss = model(X, training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    hessian = []
    for grad in gradients:
        if grad is not None:
            with tf.GradientTape() as tape2:
                tape2.watch(model.trainable_variables)
                hessian_row = tape2.gradient(grad, model.trainable_variables)
            hessian.append(hessian_row)
        else:
            hessian.append(None)
    return hessian


# In[11]:


#我们需要确保只处理有效的Hessian矩阵元素。
def check_critical_point(hessian_matrix):
    hessian_matrix_flat = []
    for hessian_row in hessian_matrix:
        if hessian_row is not None:
            for elem in hessian_row:
                if elem is not None:
                    hessian_matrix_flat.append(elem.numpy().flatten())
    
    if len(hessian_matrix_flat) == 0:
        return "Other"
    
    hessian_matrix_flat = np.concatenate(hessian_matrix_flat)
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(hessian_matrix_flat)
    if np.all(eigenvalues > 0):
        return "Local Minimum"
    elif np.all(eigenvalues < 0):
        return "Local Maximum"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Saddle Point"
    else:
        return "Other"

# 确保输入数据和模型的dtype一致
X_train = X_train.astype(np.float64)

# 对所有训练数据计算Hessian矩阵并判断极值类型
results = []
for i in range(len(X_train)):
    hessian_matrix = get_hessian(model, X_train[i:i+1])
    result = check_critical_point(hessian_matrix)
    results.append(result)


# In[12]:


# 统计各类极值点的数量
local_min_count = results.count("Local Minimum")
local_max_count = results.count("Local Maximum")
saddle_point_count = results.count("Saddle Point")
other_count = results.count("Other")

# 可视化结果
labels = ['Local Minimum', 'Local Maximum', 'Saddle Point', 'Other']
counts = [local_min_count, local_max_count, saddle_point_count, other_count]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['green', 'red', 'blue', 'gray'])
plt.xlabel('Point Type')
plt.ylabel('Count')
plt.title('Distribution of Critical Points in the Training Data')
plt.show()


# In[ ]:




