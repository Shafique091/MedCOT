# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel, AutoModel

##############################
# Cross-Attention Block
##############################

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
       
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
       
        attn_output, attn_weights = self.attn(x, x, x)
        out = self.norm(x + attn_output)
        return out, attn_weights


##############################
# Mixture of Experts Module
##############################

class Expert(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_dim, num_experts=3, top_k=3):
        super().__init__()
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):

        gate_scores = F.softmax(self.gate(x), dim=-1)  # [B, num_experts]
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        expert_outputs = []
   
        for b in range(x.size(0)):
            out_b = 0
            for k in range(self.top_k):
                expert_idx = topk_indices[b, k]
                expert = self.experts[expert_idx]
                weighted = topk_scores[b, k] * expert(x[b].unsqueeze(0))
                out_b = out_b + weighted
            expert_outputs.append(out_b)
        aggregated = torch.cat(expert_outputs, dim=0)  
        return aggregated, gate_scores  


##############################
# MedCoT Model
##############################

class MedCoTModel(nn.Module):
    def __init__(
        self,
        t5_model_name="google/flan-t5-base",
        medvit_model_name="ucl-med/medclip-vit-base-patch16",  
        hidden_dim=512,
        num_experts=4,
        moe_topk=2
    ):
        super().__init__()

        self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name)

        self.txt_proj = nn.Linear(self.text_encoder.config.d_model, hidden_dim)
        self.image_encoder = AutoModel.from_pretrained(medvit_model_name)
        self.img_proj = nn.Linear(self.image_encoder.config.hidden_size, hidden_dim)
        

        self.cross_attn = CrossAttentionBlock(hidden_dim)

        self.moe = MixtureOfExperts(hidden_dim, num_experts, moe_topk)
        

        self.project_to_t5 = nn.Linear(hidden_dim, self.text_encoder.config.d_model)
        

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
          images: raw image tensor (B, C, H, W) for Med-ViT
          input_ids: tokenized input text [B, seq_len]
          attention_mask: attention mask for text [B, seq_len]
          labels: tokenized labels for generation (optional) [B, target_seq_len]
        """
        # ----- Extract Text Embeddings via FLAN-T5 Encoder -----
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = text_outputs.last_hidden_state  
        txt_emb = self.txt_proj(txt_emb) 

        # ----- Extract Image Embeddings via Med-ViT -----
        img_outputs = self.image_encoder(images)
     
        img_cls = img_outputs.last_hidden_state[:, 0, :]  
        img_emb = self.img_proj(img_cls).unsqueeze(1)  

        # ----- Fusion: Concatenate Image and Text Embeddings -----
     
        fusion_input = torch.cat([img_emb, txt_emb], dim=1)  # [B, 1 + seq_len, hidden_dim]
        
        # Apply cross-attention over the concatenated tokens.
        fused_out, attn_weights = self.cross_attn(fusion_input)
     
        fused_vector = fused_out[:, 0, :] 

        # ----- Mixture of Experts on the Fused Representation -----
        moe_out, gating_scores = self.moe(fused_vector) 

        # ----- Project to T5 Input Embedding Space -----
        prefix_embed = self.project_to_t5(moe_out).unsqueeze(1)  

        # ----- T5 Decoder for Answer Generation -----
 
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
      
        encoder_hidden_states = torch.cat([prefix_embed, encoder_outputs.last_hidden_state], dim=1)
      
        extended_mask = F.pad(attention_mask, (1, 0), value=1)

        outputs = self.t5(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=extended_mask,
            labels=labels,
            return_dict=True
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "attn_weights": attn_weights,   
            "gating_scores": gating_scores    
        }
