from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建一个带有重试逻辑的requests session
def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


def fetch_protein_function(session, uniprot_id):
    try:
        url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}'
        response = session.get(url)

        if response.status_code == 200:
            data_json = response.json()
            function_info = next(
                (item['texts'][0]['value'] for item in data_json['comments'] if item['commentType'] == 'FUNCTION'),
                '未找到功能信息')
            return uniprot_id, function_info, None
        else:
            error_message = f"错误：无法获取UniProt ID {uniprot_id} 的信息，状态码: {response.status_code}"
            return uniprot_id, '未找到信息', error_message

    except Exception as e:
        error_message = f"异常：处理UniProt ID {uniprot_id} 时出现错误: {str(e)}"
        return uniprot_id, '异常发生', error_message


data_path = '../data/GSE225349_somalogic_protein_ids_validated.csv'
data = pd.read_csv(data_path)
session = create_session()
results = []
errors = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_protein_function, session, uniprot_id): uniprot_id for uniprot_id in
               data['UniProt']}
    for future in as_completed(futures):
        uniprot_id, function_info, error = future.result()
        results.append((uniprot_id, function_info))
        if error:
            errors.append((uniprot_id, error))

# 处理结果和错误信息
results_df = pd.DataFrame(results, columns=['UniProt', 'Protein Function'])
data = data.merge(results_df, on='UniProt')
data.to_csv('updated_data.csv', index=False)

if errors:
    errors_df = pd.DataFrame(errors, columns=['UniProt', 'Error'])
    errors_df.to_csv('errors.csv', index=False)
