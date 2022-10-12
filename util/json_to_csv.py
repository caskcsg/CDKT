from types import DynamicClassAttribute
import pandas as pd
import json
import csv


def trans(jsonpath, csvpath):
    json_file = open(jsonpath, 'r', encoding='utf8')
    csv_file = open(csvpath, 'w', newline='')
    keys = []
    writer = csv.writer(csv_file)
    json_data = json_file.readline()
    dic_data = json.loads(json_data, encoding='utf8')
    keys = dic_data.keys()
    writer.writerow(keys)
    

    for line in json_file.readlines():
        dic = json.loads(line)
        writer.writerow(dic.values())
    json_file.close()
    csv_file.close()

if __name__ == '__main__':
    trans('/workspace/hateful_memes/TLM/train_twitter.jsonl', '/workspace/hateful_memes/TLM/train_twitter.csv')



# with open('/workspace/hateful_memes/TLM/train_hateful.jsonl','r',encoding='utf8')as fp:
#     a = []
#     keys = []
#     writer = csv.writer('/workspace/twitter/ALBEF/train.csv')
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/train.csv','w',encoding='utf8')as fp:
#             csv.writer(a, fp)

# with open('/workspace/hateful_memes/TLM/train_twitter.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/hateful_memes/TLM/train_twitter.csv','w',encoding='utf8')as fp:
#             json.dump(a, fp)