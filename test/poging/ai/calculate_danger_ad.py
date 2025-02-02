import numpy as np
from sklearn.naive_bayes import GaussianNB

# 初始化贝叶斯模型
def train_bayesian_model(training_data, labels):
    """
    训练贝叶斯模型
    :param training_data: 训练数据 (numpy array)
    :param labels: 数据对应的危险等级标签
    :return: 训练好的贝叶斯模型
    """
    model = GaussianNB()
    model.fit(training_data, labels)
    return model

# 危险值评估函数
def evaluate_danger_factors(factors, weights, n=2, bayesian_model=None):
    """
    评估危险值
    :param factors: dict, 各个危险因素及其得分值。例如：
        {
            "age": 0.8,         # 年龄得分
            "weather": 0.7,     # 天气得分
            "time_of_day": 0.5, # 时间得分
            "fall_duration": 0.9, # 摔倒时长
        }
    :param weights: dict, 各个因素的权重，例如：
        {
            "age": 0.4,
            "weather": 0.3,
            "time_of_day": 0.2,
            "fall_duration": 0.5,
        }
    :param n: int, 非线性放大系数，默认值为2。
    :param bayesian_model: 已训练的贝叶斯模型（可选）。若提供，进行二次预测校正。
    :return: float, 最终的危险值。
    """
    # 确保 factors 和 weights 的键一致
    assert set(factors.keys()) == set(weights.keys()), "Factors and weights must have the same keys."

    # 计算危险值 Σ(因素得分 × 权重)
    danger_score = sum(factors[key] * weights[key] for key in factors.keys())

    # 应用非线性放大
    danger_score = danger_score ** n

    # 使用贝叶斯模型校正（如果提供）
    if bayesian_model:
        factor_array = np.array([list(factors.values())])  # 转换为适合模型的输入格式
        bayesian_prediction = bayesian_model.predict_proba(factor_array)[0]
        # 加权平均校正危险值
        danger_score = 0.7 * danger_score + 0.3 * bayesian_prediction[1]

    return danger_score

# 示例用法
def main():
    # 假设的训练数据
    training_data = np.array([
        [0.8, 0.7, 0.5, 0.9],  # 样本1的特征值
        [0.6, 0.5, 0.3, 0.8],  # 样本2的特征值
        [0.4, 0.2, 0.7, 0.6],  # 样本3的特征值
    ])
    labels = np.array([1, 0, 0])  # 样本的危险等级（1=高危, 0=低危）

    # 训练贝叶斯模型
    bayesian_model = train_bayesian_model(training_data, labels)

    # 实际评估输入
    factors = {
        "age": 0.8,
        "weather": 0.7,
        "time_of_day": 0.5,
        "fall_duration": 0.9,
    }
    weights = {
        "age": 0.4,
        "weather": 0.3,
        "time_of_day": 0.2,
        "fall_duration": 0.5,
    }

    # 计算危险值
    danger_value = evaluate_danger_factors(factors, weights, n=2, bayesian_model=bayesian_model)
    print(f"Calculated Danger Value: {danger_value:.2f}")

if __name__ == "__main__":
    main()
