import pandas as pd
import requests

def fetch_protein_function(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        # 尝试从评论中获取功能描述
        function_section = next((item for item in data.get('comments', []) if item['commentType'] == 'FUNCTION'), None)
        if function_section:
            function_text = function_section['texts'][0]['value']
        else:
            # 如果没有找到评论中的功能描述，尝试从蛋白描述中获取
            protein_description = data.get('proteinDescription', {})
            recommended_name = protein_description.get('recommendedName', {})
            submission_name = protein_description.get('submissionNames', [{}])[0]

            if 'fullName' in recommended_name:
                function_text = recommended_name['fullName'].get('value', 'No function description available')
            elif 'fullName' in submission_name:
                function_text = submission_name['fullName'].get('value', 'No function description available')
            else:
                function_text = "No function description available"

        return function_text
    else:
        return "Function not found"


# 读取CSV文件
data = pd.read_csv('../data/GSE225349_somalogic_protein_ids_validated.csv')  # 请替换为您的文件路径

# 对每个UniProt ID查询功能描述
data['Function'] = data['UniProt'].apply(fetch_protein_function)

# 保存更新后的文件
data.to_csv('updated_file.csv', index=False)  # 您可以自定义输出文件名
