from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F



class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)#ve是三分类需要改成二分类
                )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)#同上
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, input_ids, attention_mask, prompt_label_idx, labels, alpha=0, train=True):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        
        batch_size = input_ids.shape[0]

        if train:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )  
            # print(output.last_hidden_state[0,prompt_label_idx[2]].shape)
            # print(self.text_encoder.embeddings.word_embeddings.weight.transpose(1,0).shape)
            # torch.Size([1, 768])
            # torch.Size([768, 30522])

            prediction = [torch.mm(output.last_hidden_state[i,prompt_label_idx[i]], self.text_encoder.embeddings.word_embeddings.weight.transpose(1,0)) for i in range (batch_size)] 
            prediction = torch.cat(prediction, 0)        
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(input_ids, 
                                               attention_mask = attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = [torch.mm(output.last_hidden_state[i,prompt_label_idx[i]], self.text_encoder.embeddings.word_embeddings.weight.transpose(1,0)) for i in range (batch_size)] 
                    prediction_m = torch.cat(prediction_m, 0)        

                labels = labels.squeeze(1)
                loss = (1-alpha)*F.cross_entropy(prediction, labels) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                labels = labels.squeeze(1)
                # print(labels.shape) #[24]
                # print(prediction.shape) #[24, 30522]
                loss = F.cross_entropy(prediction, labels)                
            return loss, prediction
            
        else:
            output = self.text_encoder(input_ids, 
                                       attention_mask = attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            prediction = [torch.mm(output.last_hidden_state[i,prompt_label_idx[i]], self.text_encoder.embeddings.word_embeddings.weight.transpose(1,0)) for i in range (batch_size)] 
            prediction = torch.cat(prediction, 0)        
                      
            return prediction
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

