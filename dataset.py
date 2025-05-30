from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class VQARADDataset(Dataset):
    def __init__(self, excel_path, image_dir, tokenizer, answer_vocab=None, transform=None):
        self.df = pd.read_excel(excel_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab or self.build_vocab()
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def build_vocab(self):
        unique_answers = self.df["ANSWER"].astype(str).str.lower().unique().tolist()
        return {ans: idx for idx, ans in enumerate(unique_answers)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"VQARAD/VQA_RADImageFolder/{row['IMAGEID']}"
        question = str(row["QUESTION"])
        answer = str(row["ANSWER"]).lower()

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize question
        q_enc = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        q_input_ids = q_enc['input_ids'].squeeze(0)
        q_attn_mask = q_enc['attention_mask'].squeeze(0)

        # Map answer to class index
        answer_idx = torch.tensor(self.answer_vocab[answer], dtype=torch.long)

        return image, q_input_ids, q_attn_mask, answer_idx
