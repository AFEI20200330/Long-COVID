import pandas as pd
import requests


def fetch_uniprot_function(uniprot_id):
    """根据UniProt ID查询功能信息"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            function_info = next(
                (item['texts'][0]['value'] for item in data['comments'] if item['commentType'] == 'FUNCTION'),
                None)
            return function_info
        else:
            return None
    except:
        return None


# 指定文件路径
file_path = 'Non_English_Protein_Functions.csv'
data = pd.read_csv(file_path)
results = []

for index, row in data.iterrows():
    if type( row['UniProt']) == float:
        continue
    uniprots = row['UniProt'].split()
    functions = []

    if len(uniprots) == 1:
        functions.append(None)  # 只有一个UniProt ID的情况
    else:
        for uni in uniprots:
            func = fetch_uniprot_function(uni)
            functions.append(func if func else "No function found")

    results.append({
        'SeqId': row['SeqId'],
        'Target': row['Target'],
        'UniProt': row['UniProt'],
        'Functions': "; ".join(filter(None, functions)) if functions else None
    })

# 将结果保存为新的Excel文件
results_df = pd.DataFrame(results)
output_path = file_path.replace('.csv', '_updated.xlsx')
results_df.to_excel(output_path, index=False)
print(f"Updated file saved as {output_path}")
