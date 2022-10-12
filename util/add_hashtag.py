import json
import shutil
a = []
with open('/workspace/hateful_memes/train.jsonl','r',encoding='utf8')as fp:
    for line in fp.readlines():
        dic = json.loads(line)
        dicc = {}
        with open('/workspace/hateful_memes/annotations/train.json','r',encoding='utf8')as tp:
            dicc["image"] = dic["id"]
            dicc["sentence"] = dic["text"] 
            dicc["label"] = dic["label"]
            a.append(dicc)
    # with open('/workspace/twitter/ALBEF/test.json','w',encoding='utf8')as fp:
    #         json.dump(a, fp)
b = []
with open('/workspace/hateful_memes/train.jsonl','r',encoding='utf8')as fp:
    for line in fp.readlines():
        dic = json.loads(line)
        dicc = {}
        with open('/workspace/hateful_memes/annotations/train.json','r',encoding='utf8')as tp:
            dicc["image"] = dic["id"]
            if dic["id"] == '1':
                dicc["sentence"] = dic["text"] + '#'
            dicc["label"] = dic["label"]
            a.append(dicc)
    with open('/workspace/twitter/ALBEF/test.json','w',encoding='utf8')as fp:
            json.dump(a, fp)

    

# with open('/workspace/twitter/val.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/val.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)
# with open('/workspace/hateful_memes/dev_seen.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/hateful_memes/ALBEF/dev_seen.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)
# with open('/workspace/twitter/test.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/test.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)

# with open('/workspace/hateful_memes/test_seen.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/hateful_memes/ALBEF/test_seen.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)
