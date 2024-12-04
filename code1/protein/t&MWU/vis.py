import re

import pandas as pd
from scipy import stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt

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

    # 判断是否符合正态分布（通常p值>0.05可以认为符合正态分布）
    if p_norm_0 > 0.05 and p_norm_1 > 0.05:
        # 如果两组数据都符合正态分布，则进行t检验
        stat, p_value = stats.ttest_ind(group_0, group_1, nan_policy='omit')
        test_method = 't检验'
    else:
        # 如果有任何一组数据不符合正态分布，则进行Mann-Whitney U检验
        stat, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
        test_method = 'Mann-Whitney U检验'

    # 判断是否显著差异
    significant = '是' if p_value < 0.05 else '否'

    # 存储结果
    analysis_results.append([protein, test_method, p_value, significant])

# 将结果转换为DataFrame
analysis_results_df = pd.DataFrame(analysis_results, columns=['Protein', 'Test Method', 'p_value', 'Significant'])

# 筛选出显著差异的蛋白质
significant_proteins = analysis_results_df[analysis_results_df['Significant'] == '是']['Protein']

# 提取显著差异蛋白质的数据
significant_data_0 = cluster_0[significant_proteins]
significant_data_1 = cluster_1[significant_proteins]

# 每次绘制最多6个蛋白质的箱线图
max_proteins_per_plot = 6
num_plots = (len(significant_proteins) // max_proteins_per_plot) + 1

# 可视化差异显著的蛋白质，每6个蛋白质一张图
for plot_index in range(num_plots):
    plt.figure(figsize=(15, 5))
    start_idx = plot_index * max_proteins_per_plot
    end_idx = min((plot_index + 1) * max_proteins_per_plot, len(significant_proteins))

    # 为每张图命名
    protein_names = '_'.join(significant_proteins[start_idx:end_idx])
    # 替换掉不合法字符（如逗号、斜杠等）
    safe_protein_names = re.sub(r'[\\/*?:"<>|,]', '_', protein_names)

    for i, protein in enumerate(significant_proteins[start_idx:end_idx]):
        plt.subplot(2, 3, i + 1)
        plt.boxplot([significant_data_0[protein], significant_data_1[protein]], labels=['Cluster 0', 'Cluster 1'])
        plt.title(protein)
        plt.ylabel('Protein Expression')

    # 保存图像文件
    plot_file_name = f"res/{safe_protein_names}_comparison.jpg"
    plt.tight_layout()
    plt.savefig(plot_file_name,dpi=700)

    # 显示图像
    plt.show()