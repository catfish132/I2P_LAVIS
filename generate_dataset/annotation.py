import os
import csv
import json
import urllib.request

# 创建images文件夹
if not os.path.exists('images'):
    os.makedirs('images')

# 读取CSV文件
with open('room_3vj.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # 初始化输出的json数组
    output_json = []
    sum=0
    for row in reader:
        if sum<1200:
            sum+=1
            continue
        image_url = row['room_image-src']
        if image_url.startswith('//'):
            image_url = 'https:' + image_url
        # 下载图片到images文件夹，图片的名称为web-scraper-order的值
        urllib.request.urlretrieve(image_url, os.path.join('images', row['\ufeffweb-scraper-order'] + '.jpg'))

        # 创建一个新的字典，id为web-scraper-order的值，image_path为图片的路径
        image_dict = {}
        image_dict['id'] = row['\ufeffweb-scraper-order']
        image_dict['image_path'] = os.path.join('images', row['\ufeffweb-scraper-order'] + '.jpg')

        # 将字典添加到json数组中
        output_json.append(image_dict)

# 将json数组写入到文件中
with open('output.json', 'w') as jsonfile:
    json.dump(output_json, jsonfile)
