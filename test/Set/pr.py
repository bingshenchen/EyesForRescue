from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# 示例数据：真实标签和预测概率
true_labels = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  # 0 = Fine, 1 = Need Help
predicted_probabilities = [0.1, 0.9, 0.8, 0.2, 0.7, 0.1, 0.95, 0.3, 0.4, 0.85]  # 模型预测概率

# 计算 Precision, Recall 和阈值
precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probabilities)

# 计算 AUC (曲线下面积)
pr_auc = auc(recall, precision)

# 绘制 Precision-Recall 曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})', color='b')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
