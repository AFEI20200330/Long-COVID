import pandas as pd
from scipy.stats import shapiro, mannwhitneyu, ttest_ind

# 加载数据
data_path = '../clinical_data_labeled_cluster.xlsx'  # 替换为您的文件路径
data = pd.read_excel(data_path)

# 识别并编码文本数据
text_columns = [col for col in data.columns if data[col].dtype == 'object']
encoded_data = data.copy()
for column in text_columns:
    encoded_data[column], _ = pd.factorize(encoded_data[column])

# 分割数据为不同的聚类
encoded_cluster_0 = encoded_data[encoded_data['Cluster'] == 0]
encoded_cluster_1 = encoded_data[encoded_data['Cluster'] == 1]

# 对所有特征进行统计分析
full_encoded_results = {}
for feature in encoded_data.columns[3:]:  # 跳过前三列（ID等非特征列）
    values_cluster_0 = encoded_cluster_0[feature].dropna()
    values_cluster_1 = encoded_cluster_1[feature].dropna()

    if len(values_cluster_0) >= 3 and len(values_cluster_1) >= 3 and values_cluster_0.var() > 0 and values_cluster_1.var() > 0:
        # 正态性检验
        normality_cluster_0 = shapiro(values_cluster_0)
        normality_cluster_1 = shapiro(values_cluster_1)

        # 根据正态性选择统计测试
        if normality_cluster_0.pvalue > 0.05 and normality_cluster_1.pvalue > 0.05:
            test_stat, p_value = ttest_ind(values_cluster_0, values_cluster_1)
            test_used = 'T-test'
        else:
            test_stat, p_value = mannwhitneyu(values_cluster_0, values_cluster_1)
            test_used = 'Mann-Whitney U'

        full_encoded_results[feature] = {
            'Test Used': test_used,
            'P-Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        }

# 将结果转换为DataFrame并保存
full_encoded_results_df = pd.DataFrame(full_encoded_results).transpose()
results_path = 'full_encoded_cluster_analysis_results.xlsx'
full_encoded_results_df.to_excel(results_path)

