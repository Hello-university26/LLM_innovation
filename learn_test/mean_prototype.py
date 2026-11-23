import numpy as np

# 1. 创建三个音乐类别的样本向量
music_vectors = np.array([
    [0.9, 0.2, 0.1],
    [0.8, 0.3, 0.1],
    [0.9, 0.1, 0.2]
])

# 2. 计算三个向量的平均值
prototype = np.mean(music_vectors, axis=0)

# 3. 输出类别原型
print("音乐类别的原型：")
print(prototype)
