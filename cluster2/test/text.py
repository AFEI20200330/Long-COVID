import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 加载数据
df = pd.read_excel('../filtered_output.xlsx')
# 假设数据框df中除了'cluster'列，其余列都是蛋白质的测量数据
protein_columns = df.columns.drop('Target','Cluster')
# 用于存储分析结果的数据框
results = pd.DataFrame(columns=['Protein', 'Test', 'p-value', 'Is Normal Distribution','Significant'])

# 遍历每个蛋白质
for protein in protein_columns:
    # 由于蛋白质名称可能包含特殊字符或空格，我们在模型中使用Q()来安全地引用列名
    formula = f'Q("{protein}") ~ C(Cluster)'
    # 数据正态性检验
    if all(stats.shapiro(df[df['Cluster'] == i][protein])[1] > 0.05 for i in range(4)):
        # 如果所有组的数据都通过Shapiro-Wilk正态性检验
        # 执行ANOVA
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table['PR(>F)'][0]
        significant = 'Yes' if p_value < 0.05 else 'No'
        new_row = pd.DataFrame({
            'Protein': [protein], 'Test': ['ANOVA'], 'p-value': [p_value],
            'Is Normal Distribution': [True], 'Significant': [significant]
        })
        results = pd.concat([results, new_row], ignore_index=True)
    else:
        # 如果数据不符合正态分布，执行Kruskal-Wallis非参数检验
        kruskal = stats.kruskal(*[df[df['Cluster'] == i][protein].dropna() for i in range(4)])
        p_value = kruskal.pvalue
        significant = 'Yes' if p_value < 0.05 else 'No'
        new_row = pd.DataFrame({
            'Protein': [protein], 'Test': ['Kruskal-Wallis'], 'p-value': [p_value],
            'Is Normal Distribution': [False], 'Significant': [significant]
        })
        results = pd.concat([results, new_row], ignore_index=True)

# 输出结果
print(results)

# 可以选择保存结果到Excel文件
results.to_excel('analysis_results.xlsx', index=False)
