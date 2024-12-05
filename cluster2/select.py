import pandas as pd

# 加载两个Excel文件
df1 = pd.read_excel('clinical_data_labeled_cluster.xlsx')
df2 = pd.read_excel('ONLY.xlsx')


# 修改1.xlsx的数据框，仅保留Blood_Sample_ID和cluster列，并去除重复
df1 = df1[['Blood_Sample_ID', 'Cluster']].drop_duplicates()

# 在2.xlsx中筛选出Target列在1.xlsx的Blood_Sample_ID中的所有行
# 同时使用merge函数添加cluster信息
filtered_df = pd.merge(df2, df1, left_on='Target', right_on='Blood_Sample_ID', how='inner')

# 输出结果到新的Excel文件
filtered_df.to_excel('filtered_output.xlsx', index=False)