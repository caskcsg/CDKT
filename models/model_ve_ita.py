from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertForMaskedLM

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
        ###
        embed_dim = config['embed_dim']  #256
        vision_width = config['vision_width']#768


        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        ###
        self.temp = nn.Parameter(torch.ones([]) * config['temp']) #0.07
        text_width = self.text_encoder.config.hidden_size #768
        self.vision_proj = nn.Linear(vision_width, embed_dim) #[768, 256]
        self.text_proj = nn.Linear(text_width, embed_dim)  #[768, 256]
        self.queue_size = config['queue_size'] #65536

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )

        # # create momentum models
        # self.visual_encoder_m = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        # self.text_proj_m = nn.Linear(text_width, embed_dim)
        # self.cls_head_m = nn.Sequential(
        #               nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
        #               nn.ReLU(),
        #               nn.Linear(self.text_encoder.config.hidden_size, 2)#同上
        #             )
        # self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                         [self.text_encoder,self.text_encoder_m],
        #                         [self.cls_head,self.cls_head_m],
        #                        ]
        # self.copy_params() 

        ### create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        

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
        
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
            
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        
        if train:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            prediction = self.cls_head(output.last_hidden_state[:,0,:])  

            ###                
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1) #256
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')#输出第6层
            text_embeds = text_output.last_hidden_state 
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1) #256

            if self.distill:                
                with torch.no_grad():
                    self.temp.clamp_(0.001,0.5)
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:]) 


                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
                    image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
                    text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                        return_dict = True, mode = 'text')    
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
                    text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp 
                    sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                    sim_targets.fill_diagonal_(1)  
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets 
                sim_i2t = image_feat @ text_feat_all / self.temp 
                sim_t2i = text_feat @ image_feat_all / self.temp 
                loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
                loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
                loss_ita = (loss_i2t+loss_t2i)/2  

                loss_fc = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss_fc = F.cross_entropy(prediction, targets)                
            return loss_fc, loss_ita
            
        else:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            prediction = self.cls_head(output.last_hidden_state[:,0,:])                        
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
                

