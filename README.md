# MedCoT: Medical Visual Question Answering with Mixture of Experts and Flan-T5

This repository is being maintaned for reasearch on the **MedCoT** model as described in the [MedCoT paper](https://github.com/JXLiu-AI/MedCoT). MedCoT takes a medical image and a question as input, fuses the information using a cross-attention network and a Mixture of Experts model, and generates answers using a Flan-T5 generative decoder. Additionaly saliency map is also extracted for better model explainablity, future work includes to fuse the explainability method to train the attention network based on the saliency map. 

---

## Features

- Image feature extraction using Med-ViT  (Freezed) 
- Text feature extraction using Flan-T5 tokenizer and embeddings  (Freezed) 
- Cross-Attention Encoder for multimodal fusion  (Trained)
- Mixture of Experts network for feature aggregation  
- Flan-T5 generative decoder for answer generation(Freezed)  
- Multiple explainability methods: attention maps, gating logs, saliency heatmaps  
- Evaluation on VQA-RAD dataset with BLEU, ROUGE, and yes/no accuracy metrics  

---

## Directory Structure

C:\Users\Shafique\MedCoTVQARad  
├── VQA_RADImageFolder/ # Medical images for VQA-RAD test set  
├── SLAKE/
├── VQA_RADDatasetPublic.xlsx     # VQA-RAD question-answer pairs  
├── yesno_questions.csv           
├── medcot_epoch_final.pth         
├── dataset.py                    
├── model.py                      
├── train.py                      
├── test.py                       
├── feature_extract.py           
├── README.md                     

---
Usage
Dataset
SlakeDataset: Used for training on SLAKE-EN dataset (JSON annotation format).

VQARADDataset: Used for testing on VQA-RAD dataset (Excel annotations).

## Training

python train.py
Loads SLAKE-EN dataset, extracts image and text embeddings.

Trains the MedCoT model.

Saves model checkpoints.

## Evaluation
python test.py
Loads VQA-RAD dataset.

Loads trained model checkpoint.

Runs inference and computes BLEU, ROUGE, and yes/no question accuracy.

Prints sample predictions.

## Model Architecture
Input: Image embeddings (Med-ViT) and question embeddings (Flan-T5).

Projection Layer: Projects in 512d.

Cross-Attention Encoder: Multimodal fusion via multi-head attention.

Mixture of Experts: Aggregates fused features with gating for 3 experts.(top-k originally)

Projection Layer: Projects features to Flan-T5 embedding dimension.

Flan-T5 Decoder(Freezed): Generates textual answer.

### Explainability: (Attention maps, gating logs)(Not Included), gradient-based saliency(Included in the Model 1 Notebook file).
### 
