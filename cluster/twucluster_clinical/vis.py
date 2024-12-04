import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 加载数据
file_path = 'clinical_data_labeled_cluster.xlsx'  # 请替换为你的数据文件路径
data = pd.read_excel(file_path)

# 获取数据的列名，排除前3列
columns_to_analyze = data.columns[3:]


# 函数：根据条件创建饼图或柱状图
# 函数：根据条件创建饼图或柱状图
def create_charts_with_order(data, column, cluster_0, cluster_1):
    if data[column].dropna().empty and cluster_0[column].dropna().empty and cluster_1[column].dropna().empty:
        return  # 如果该列没有数据，则跳过

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Distribution of {column} for Clusters 0 and 1', fontsize=16)

    # 如果是数值型数据，使用并列柱状图，并按从小到大的顺序排列
    # 获取两类数据的频率
    cluster_0_counts = cluster_0[column].dropna().value_counts(normalize=True).sort_index()
    cluster_1_counts = cluster_1[column].dropna().value_counts(normalize=True).sort_index()

    # 合并 cluster_0 和 cluster_1 的索引，确保两者有相同的类别
    all_index = sorted(set(cluster_0_counts.index).union(set(cluster_1_counts.index)))
    cluster_0_counts = cluster_0_counts.reindex(all_index, fill_value=0)
    cluster_1_counts = cluster_1_counts.reindex(all_index, fill_value=0)

    # 计算柱状图的宽度
    width = 0.35  # 每个柱体的宽度
    x = range(len(all_index))  # 横轴位置

    # 绘制并列柱状图
    ax.bar(x, cluster_0_counts, width, color='#FF9999', label='Cluster 0')
    ax.bar([p + width for p in x], cluster_1_counts, width, color='#66B3FF', label='Cluster 1')

    ax.set_title('Cluster 0 vs Cluster 1')
    ax.set_xlabel(column)
    ax.set_ylabel('Proportion')

    # 设置横轴的数值格式为保留一位小数
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(all_index)

    # 添加图例
    ax.legend()

    ax.set_axisbelow(True)
    ax.grid(False)  # 移除网格线

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应标题
    plt.savefig(f'res/{column}_cluster_comparison.jpg',dpi=700)  # 保存每个图表为png文件
    plt.show()


# 创建 cluster 0 和 cluster 1 的数据子集
cluster_0 = data[data['Cluster'] == 0]
cluster_1 = data[data['Cluster'] == 1]

# 遍历所有指标并进行可视化
for column in columns_to_analyze:
    create_charts_with_order(data, column, cluster_0, cluster_1)
