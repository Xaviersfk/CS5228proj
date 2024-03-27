import csv

txt_file_path = '..\\raw_data\\xtrain.txt'
csv_file_path = '..\\raw_data\\xtrain.csv'

with open(txt_file_path, 'r', encoding='utf-8') as txt_file, \
     open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    
    for line in txt_file:
        label, text = line.strip().split('\t', 1)
        csv_writer.writerow([label, text])

print(f'文件已转换完成，保存为：{csv_file_path}')
