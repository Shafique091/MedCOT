import torch
import torch.nn as nn
from transformers import ViTModel, T5EncoderModel, AutoTokenizer

class Expert(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, output_dim))

    def forward(self, x):
        return self.fc(x)

class MoEDecoder(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, output_dim=512):
        super().__init__()
        self.experts = nn.ModuleList([Expert(hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, fused_features):
        gate_scores = torch.softmax(self.gate(fused_features), dim=-1)
        expert_outputs = torch.stack([expert(fused_features) for expert in self.experts], dim=1)
        out = (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return out

class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.moe_decoder = MoEDecoder(768)
        self.classifier = nn.Linear(512, 1000)  # 1000 = vocab size or number of possible answers

    def forward(self, image, question_input_ids, question_attention_mask):
        vis_feat = self.vision_encoder(pixel_values=image).last_hidden_state[:, 0]  # CLS token
        text_feat = self.text_encoder(input_ids=question_input_ids.squeeze(1),
                                      attention_mask=question_attention_mask.squeeze(1)).last_hidden_state
        attn_output, _ = self.cross_attention(vis_feat.unsqueeze(1), text_feat, text_feat)
        fused = attn_output.squeeze(1)
        moe_out = self.moe_decoder(fused)
        logits = self.classifier(moe_out)
        return logits
