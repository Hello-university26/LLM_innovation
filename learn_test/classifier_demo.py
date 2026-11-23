
import numpy as np


# ===================================
# 第一部分：余弦相似度函数
# ===================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)

    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("必须输入一维向量")

    if a.shape != b.shape:
        raise ValueError("向量维度不一致")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("向量包含非法数值")

    norm_a = np.linalg.norm(a)

    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("不允许输入零向量")

    dot_product = np.dot(a, b)

    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)


# ===================================
# 第二部分：建立原型库
# ===================================

prototypes = {

    "music_play": [0.9, 0.2, 0.1],

    "weather_query": [0.1, 0.9, 0.2],

    "alarm_set": [0.1, 0.2, 0.9]

}


# ===================================
# 第三部分：输入查询向量
# ===================================

query = list(map(int,input().split()))


# ===================================
# 第四部分：计算所有类别的相似度
# ===================================

scores = {}

for intent_id, prototype in prototypes.items():
    try:
        score = cosine_similarity(query, prototype)

        scores[intent_id] = score
    except ValueError as e:
        print("计算失败：",e)
        break;



# ===================================
# 第五部分：选择最相似的类别
# ===================================

best_intent = max(scores, key=scores.get)


# ===================================
# 第六部分：输出结果
# ===================================

print("查询向量：", query)

print("\n各类别相似度：")

for intent_id, score in scores.items():

    print(intent_id, ":", round(score, 4))

print("\n最终预测类别：", best_intent)
