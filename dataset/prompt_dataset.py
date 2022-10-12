import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption



class prompt_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, temps, tokenizer, max_length = 128):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root

        self.temps = temps #{0:{}, 1:{}}
        self.get_labels(tokenizer)
        self.tokenizer = tokenizer

        self.max_length = max_length
        #self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        #此处进行改进jpg->png
        #hateful
        #image_path = os.path.join(self.image_root,'%s.png'%ann['image']) 
        # twitter  为jpg结尾     
        image_path = os.path.join(self.image_root,'%s.png'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)  

        label = str(ann['label'])
        text = ann['sentence'] 
        sent_ids = self.tokenizer.encode(text, add_special_tokens = False)
        prompt = self.temp_ids[label]['mask_ids'][0] 
        lm_label = self.temp_ids[label]['label_ids'][0]  
        input_ids = sent_ids + prompt  
        length = torch.LongTensor([len(input_ids) + 2]) 
        input_ids = torch.LongTensor([0] + input_ids + [2] + [1] * (self.max_length - length))
        attention_mask = (input_ids != 1).long()  
        lm_label = torch.LongTensor([lm_label])
        loc = length - 2

        return image, text, ann['label'], input_ids, attention_mask, lm_label, loc 


    def get_labels(self, tokenizer):
        #total = {}  
        self.temp_ids = {}
        for id in self.temps:#{'0': {'id': '0', 'template': ['it', 'is', '[MASK]'], 'label_name': 'good'}, '1': {'id': '1', 'template': ['it', 'is', '[MASK]'], 'label_name': 'bad'}}
            self.temp_ids[id] = {} 
            self.temp_ids[id]['label_ids'] = [] 
            self.temp_ids[id]['mask_ids'] = []
            temp = self.temps[id]['template']                                  
            _temp = temp.copy()
            _label = self.temps[id]['label_name']
            #{0:{'label_ids':[], 'mask_ids':[]}, }
            #{'id': '0', 'template': ['it', 'is', '<mask>'], 'label_name': 'terrible'}

            for i in range(len(_temp)):
                if _temp[i] == tokenizer.mask_token:
                    _temp[i] = _label  #将label值赋给mask  _temp = ['it', 'is', 'not-hateful']
                    _label_index = i  #记录mask位置
            
            original = tokenizer.encode(' '.join(temp), add_special_tokens = False)#将句子转化成对应模型的输入形式，默认开启
            # ['it', 'is', '[MASK]']
            
            final = tokenizer.encode(' '.join(_temp), add_special_tokens = False)
            self.temp_ids[id]['label_ids'].append(final[_label_index]) #此处追加的是mask位置的真实标签id
            
            #此处追加的是原始mask的prompt
            self.temp_ids[id]['mask_ids'].append(original)
        print(self.temp_ids)
    