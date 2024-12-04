import pandas as pd
from scipy import stats
from scipy.stats import shapiro

# 读取上传的文件
file_path = '../../merged_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 选择蛋白质列（排除前两列：Target_new, Blood_Sample_ID, Cluster）
protein_columns = df.columns[3:]

# 按Cluster分组
cluster_0 = df[df['Cluster'] == 0][protein_columns]
cluster_1 = df[df['Cluster'] == 1][protein_columns]

# 存储结果
analysis_results = []

# 对每个蛋白质进行分析
for protein in protein_columns:
    # 提取两组数据
    group_0 = cluster_0[protein].dropna()
    group_1 = cluster_1[protein].dropna()

    # 进行正态性检验（Shapiro-Wilk检验）
    _, p_norm_0 = shapiro(group_0)
    _, p_norm_1 = shapiro(group_1)

    if p_norm_0 > 0.05 and p_norm_1 > 0.05:
        # 如果两组数据都符合正态分布，则进行t检验
        stat, p_value = stats.ttest_ind(group_0, group_1, nan_policy='omit')
        test_method = 't-test'
    else:
        # 如果有任何一组数据不符合正态分布，则进行Mann-Whitney U检验
        stat, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
        test_method = 'Mann-Whitney U test'

    # 判断是否显著差异
    significant = 'True' if p_value < 0.05 else 'False'

    # 存储结果
    analysis_results.append([protein, test_method, p_value, significant])

# 将结果转换为DataFrame
analysis_results_df = pd.DataFrame(analysis_results, columns=['Protein', 'Test Method', 'p_value', 'Significant'])

# 将分析结果保存为Excel文件
output_file_path = 'cluster_comparison_results_2.xlsx'
analysis_results_df.to_excel(output_file_path, index=False)

