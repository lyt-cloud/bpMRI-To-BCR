import csv
import json

# 读取CSV文件并转换为JSON格式
csv_file = '../datasets/表格数据/test147_clinic6.csv'
json_file = '../datasets/表格数据/test147_clinic6.json'


# 打开CSV文件并读取内容
csv_data = {}
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # 读取CSV的第一行作为头部
    for row in reader:
        patient_id = row[0]
        data_values = row[1:]  # 获取除第一列外的所有列数据
        csv_data[patient_id] = data_values

# 将数据写入JSON文件
with open(json_file, 'w') as jsonfile:
    json.dump(csv_data, jsonfile, indent=4)

print(f'CSV文件 "{csv_file}" 已成功转换为JSON文件 "{json_file}"')