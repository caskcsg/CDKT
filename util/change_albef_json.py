import json
import shutil

with open('/workspace/twitter/jsonl/test.jsonl','r',encoding='utf8')as fp:
    a = []
    for line in fp.readlines():
        dic = json.loads(line)
        dicc = {}
        dicc["image"] = dic["id"]
        dicc["sentence"] = dic["text"]
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
