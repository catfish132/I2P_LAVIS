import os
import json
import re

from sklearn.model_selection import train_test_split


def read_file(file_path):
    with open(file_path, 'r', encoding='GBK') as file:
        return file.read().strip()


def keep_first_paragraph(text):
    first_paragraph = re.split('\n\n', text)[0]
    return first_paragraph


# 获取文件夹下的所有文件
image_files = os.listdir(r"D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\image")
tag_files = os.listdir(r"D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\tags")
caption_files = os.listdir(r"D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\captions")

# 获取所有数据
data = []
for image_file in image_files:
    id = os.path.splitext(image_file)[0].strip()
    tag_file = os.path.join(r"D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\tags", id + '.tag')
    caption_file = os.path.join(r"D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\captions", id + '.caption')

    # 如果tag文件和caption文件都存在，那么读取它们的内容并保存到字典中
    if os.path.isfile(tag_file) and os.path.isfile(caption_file):
        data_dict = {}
        data_dict['id'] = id
        data_dict['image_path'] = os.path.join(image_file)
        data_dict['caption'] = read_file(caption_file)
        data_dict['tags'] = keep_first_paragraph(read_file(tag_file))
        data.append(data_dict)

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# 将训练集和测试集保存到json文件中
with open(r'D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\train.json', 'w') as train_file, open(r'D:\OneDriveBackup\OneDrive\Documents\研究生\找工作\实习\巴比特实习-2022-05\data\minicoco\all\test.json', 'w') as test_file:
    json.dump(train_data, train_file)
    json.dump(test_data, test_file)
