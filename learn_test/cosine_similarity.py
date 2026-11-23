# import numpy as np
#
# a=np.array([1,2,3])
#
# b=np.array([4,5,6])
#
# dot_product=np.dot(a,b)
#
# norm_a=np.linalg.norm(a)
# norm_b=np.linalg.norm(b)
#
# similarity=dot_product/(norm_a*norm_b)
#
# print("向量a:",a)
# print("向量b:",b)
# print("点积：",dot_product)
# print("a的长度：",norm_a)
# print("b的长度：",norm_b)
# print("余弦相似度：",similarity)


import numpy as np


def cosine_similarity(a, b):

    # 转换成 NumPy 数组
    a = np.asarray(a, dtype=np.float32)

    b = np.asarray(b, dtype=np.float32)

    # 检查两个向量是不是一维
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("输入必须是一维向量")

    # 检查维度是否一致
    if a.shape != b.shape:
        raise ValueError("两个向量的维度不一致")

    # 检查是否存在非法数值
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("向量包含非法数值")

    # 计算向量长度
    norm_a = np.linalg.norm(a)

    norm_b = np.linalg.norm(b)

    # 防止零向量参与计算
    if norm_a == 0 or norm_b == 0:
        raise ValueError("不能计算零向量的余弦相似度")

    # 计算点积
    dot_product = np.dot(a, b)

    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)


# 测试
A = [1, 2, 3]

B = [4, 5, 6]

result = cosine_similarity(A, B)

print("余弦相似度：", result)
