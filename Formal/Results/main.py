import os
import threading
import time

# 定义文件夹路径
folder1_path = '../models'  # 替换为文件夹1的路径
#folder2_path = '../models_without_pca'  # 替换为文件夹2的路径

# 获取文件夹1中的所有 Python 文件
folder1_files = [f for f in os.listdir(folder1_path) if f.endswith('.py')]
folder1_files.sort()  # 如果需要按文件名排序

# # 获取文件夹2中的所有 Python 文件
# folder2_files = [f for f in os.listdir(folder2_path) if f.endswith('.py')]
# folder2_files.sort()  # 如果需要按文件名排序

# 超时时间，单位为秒，30分钟 = 1800秒
timeout = 2000

# 用于记录被强制结束的脚本路径
timeout_scripts = []


# 定义执行脚本的函数
def execute_script(py_file_path):
    start_time = time.time()
    with open(py_file_path, 'r', encoding='utf-8') as file:
        script_content = file.read()  # 读取脚本内容
        exec(script_content, locals())  # 执行脚本内容
    end_time = time.time()

    # 检查脚本是否超时
    if end_time - start_time > timeout:
        timeout_scripts.append(py_file_path)
        print(f"脚本超时，已强制结束: {py_file_path}")


# 依次执行文件夹1中的 Python 文件
for py_file in folder1_files:
    py_file_path = os.path.join(folder1_path, py_file)
    print(f"正在执行文件夹1中的脚本: {py_file_path}")
    thread = threading.Thread(target=execute_script, args=(py_file_path,))
    thread.start()
    thread.join(timeout)  # 设置超时

    if thread.is_alive():
        # 如果线程仍然在运行，表示超时，强制停止
        print(f"脚本超时，已强制结束: {py_file_path}")
        timeout_scripts.append(py_file_path)
        # 强制结束线程
        # Python中无法直接杀死线程，因此只能跳过该线程，继续执行下一个脚本
        continue

# # 依次执行文件夹2中的 Python 文件
# for py_file in folder2_files:
#     py_file_path = os.path.join(folder2_path, py_file)
#     print(f"正在执行文件夹2中的脚本: {py_file_path}")
#     thread = threading.Thread(target=execute_script, args=(py_file_path,))
#     thread.start()
#     thread.join(timeout)  # 设置超时
#
#     if thread.is_alive():
#         # 如果线程仍然在运行，表示超时，强制停止
#         print(f"脚本超时，已强制结束: {py_file_path}")
#         timeout_scripts.append(py_file_path)
#         # 强制结束线程
#         # Python中无法直接杀死线程，因此只能跳过该线程，继续执行下一个脚本
#         continue

# 输出所有被强制结束的脚本路径
if timeout_scripts:
    print("\n被强制结束的脚本如下:")
    for script in timeout_scripts:
        print(script)
else:
    print("\n没有脚本被强制结束。")
