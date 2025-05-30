# from medclip import MedCLIPModel, MedCLIPVisionModelViT
# from medclip import MedCLIPProcessor
# from medclip import PromptClassifier

# processor = MedCLIPProcessor()
# model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
# model.from_pretrained()
# clf = PromptClassifier(model, ensemble=True)
# clf.cuda()

# # prepare input image
from PIL import Image
# image = Image.open('./example_data/view1_frontal.jpg')
# inputs = processor(images=image, return_tensors="pt")

# # prepare input prompt texts
# from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
# cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
# inputs['prompt_inputs'] = cls_prompts

# # make classification
# output = clf(**inputs)
# print(output)
# {'logits': tensor([[0.5154, 0.4119, 0.2831, 0.2441, 0.4588]], device='cuda:0',
#       grad_fn=<StackBackward0>), 'class_names': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']}
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
import os
from medclip import constants

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor, PromptClassifier
from PIL import Image
import torch
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor, PromptClassifier
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

# Load weights from Hugging Face (no map_location allowed here)
model.from_pretrained("ucl-med/medclip-vit-base-patch16")

# Move model to device
model = model.to(device)

# Wrap classifier
clf = PromptClassifier(model, ensemble=True).to(device)

# Load image
image = Image.open(r'C:\Users\Rameshwar\MedCoTVQARad\MedCLIP\example_data\view1_frontal.jpg')

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

# Generate prompts
cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
inputs['prompt_inputs'] = cls_prompts

# Run inference
output = clf(**inputs)
print("Output is" )
print(output)
