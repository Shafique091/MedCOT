import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import VQARADDataset
from model import VQAModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset import VQARADDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

dataset = VQARADDataset(
    excel_path="VQARAD\VQA_RADDatasetPublic.xlsx",
    image_dir="VQARAD\VQA_RADImageFolder",  # <- change this
    tokenizer=tokenizer
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# dataset = VQARADDataset("VQARAD\VQA_RADDatasetPublic.json", "images/", tokenizer)



model = VQAModel().cuda()
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for image, input_ids, attn_mask, labels in dataloader:
        image = image.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        logits = model(image, input_ids, attn_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
