from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification



class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.config = config

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        self.cls_head_hateful = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )
        ##################改
        self.cls_head_twitter = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.cls_head_hateful[0].weight.data)
        nn.init.kaiming_normal_(self.cls_head_hateful[2].weight.data)
        nn.init.kaiming_normal_(self.cls_head_twitter[0].weight.data)
        nn.init.kaiming_normal_(self.cls_head_twitter[2].weight.data)

        classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
            "negative",
            "positive"
        ]

        promptTemplate = ManualTemplate(
            text = ["<text_a>", "It", "was", "<mask>"],
            tokenizer = tokenizer,
        )
        promptVerbalizer = ManualVerbalizer(
            classes = classes,
            label_words = {
                "negative": ["bad"],
                "positive": ["good", "wonderful", "great"],
            },
            tokenizer = tokenizer,
        )

        promptModel = PromptForClassification(
            template = promptTemplate,
            model = text_encoder,
            verbalizer = promptVerbalizer,
        )



        #不用distill
        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, 2)#同上
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 

        #print('image_embeds:', image_embeds.size()) #[20, 577, 768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        #print('image_atts:',  image_atts.size())      #[20, 577] 

        
        if train:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      ) 

            # hateful_image_embedding, twitter_image_embedding = image_embeds[:,0,:].chunk(2, 0)
            # hateful_text_embedding, twitter_text_embedding = output.hidden_states[7][:,0,:].chunk(2, 0)

            image_embed = image_embeds[:,0,:]
            #text_embed = output.hidden_states[7][:,0,:]
            text_embed = output.hidden_states[6][:,0,:]


            #print("last_hidden_state:", output.last_hidden_state.size())#[20, 32, 768]
            last_hidden_state = output.last_hidden_state[:,0,:]
            #print("last_hidden_state:", last_hidden_state.size()) #[20, 768]
            # output.hidden_states type:tuple

            # hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，len() = 13
            # 它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
            #print(output.hidden_states[7][:,0,:].size()) #output.hidden_states[1].size() torch.Size([20, 32, 768])
            
            head_hateful = self.cls_head_hateful(last_hidden_state)
            head_hateful_softmax = self.softmax(head_hateful)
            head_twitter = self.cls_head_twitter(last_hidden_state) 
            head_twitter_softmax = self.softmax(head_twitter)
            output = {
                        'last_hidden_state': last_hidden_state,
                        'head_hateful': head_hateful,
                        'head_hateful_softmax': head_hateful_softmax,
                        'head_twitter': head_twitter,
                        'head_twitter_softmax': head_twitter_softmax,
                        'image_embed': image_embed,
                        'text_embed': text_embed,
                    }

            return output
             
            # if self.distill:                
            #     with torch.no_grad():
            #         self._momentum_update()
            #         image_embeds_m = self.visual_encoder_m(image) 
            #         output_m = self.text_encoder_m(text.input_ids, 
            #                                    attention_mask = text.attention_mask, 
            #                                    encoder_hidden_states = image_embeds_m,
            #                                    encoder_attention_mask = image_atts,        
            #                                    return_dict = True
            #                                   )           
            #         prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   
            #     loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
            #         F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            # else:
            #     loss_hateful = F.cross_entropy(prediction_hateful, target_hateful)   
            #     loss_twitter = F.cross_entropy(prediction_twitter, target_twitter)                

            # return loss_hateful,  loss_twitter
            
        else:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            
            prediction_hateful = self.cls_head_hateful(output.last_hidden_state[:,0,:])          
              
            return prediction_hateful
 


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
                


