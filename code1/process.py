# @FirstAuthor: Fay
# @Coding_Time: 2024/12/3/003 下午 6:15
# @Description：
import pandas as pd

# 加载数据，这里假设CSV文件名为"data.csv"
data_path = 'updated_data.csv'
data = pd.read_csv(data_path)

# 删除重复的UniProt ID，只保留第一个出现的记录
data_unique = data.drop_duplicates(subset='UniProt', keep='first')

# 将处理后的数据保存为Excel文件
output_path = 'output.xlsx'
data_unique.to_excel(output_path, index=False)
