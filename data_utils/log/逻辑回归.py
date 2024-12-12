import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# 导入数据
data = pd.read_csv('changhaisuda2_467_bcr_clinic_log2.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 拟合逻辑回归模型
model = LogisticRegression()
model.fit(X, y)

# 计算p值
X2 = sm.add_constant(X)
est = sm.Logit(y, X2)
est2 = est.fit()
p_values = est2.pvalues[1:]

# 展示结果
results = pd.DataFrame({'Feature': X.columns, 'p-value': p_values})
results['p-value'] = results['p-value'].map('{:.4f}'.format)
print(results)