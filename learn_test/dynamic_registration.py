
import numpy as np


# ==================================
# 第一部分：创建原型库
# ==================================

prototypes = {

    "music_play": np.array([0.9, 0.2, 0.1]),

    "weather_query": np.array([0.1, 0.9, 0.2]),

    "alarm_set": np.array([0.1, 0.2, 0.9])

}


# ==================================
# 第二部分：定义动态注册函数
# ==================================

def register_intent(intent_id, prototype):

    # 检查类别是否已经存在
    if intent_id in prototypes:

        raise ValueError("这个类别已经注册过了")

    # 转换为NumPy数组
    prototype = np.asarray(
        prototype,
        dtype=np.float32
    )

    # 检查是否为一维向量
    if prototype.ndim != 1:

        raise ValueError("原型必须是一维向量")

    # 检查向量中是否存在非法数值
    if not np.isfinite(prototype).all():

        raise ValueError("原型包含非法数值")

    # 检查零向量
    norm = np.linalg.norm(prototype)

    if norm == 0:

        raise ValueError("原型不能是零向量")

    # 归一化
    prototype = prototype / norm

    # 注册新类别
    prototypes[intent_id] = prototype

    print("注册成功：", intent_id)


# ==================================
# 第三部分：注册导航类别
# ==================================

register_intent(

    "navigation_start",

    [0.2, 0.2, 0.9]

)


# ==================================
# 第四部分：查看原型库
# ==================================

print("\n当前已经注册的类别：")

for intent_id in prototypes:

    print(intent_id)
