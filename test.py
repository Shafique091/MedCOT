# test.py

import torch
from torch.utils.data import DataLoader
from dataset import VQARADDataset
from model import MedCoTModel
from transformers import T5Tokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def evaluate(model, dataloader, device, yesno_csv_path):
    model.eval()
    all_predictions = []
    all_labels = []
    all_img_names = []

    yesno_df = pd.read_csv(yesno_csv_path)
    yesno_dict = dict(zip(yesno_df['image_filename'], yesno_df['question_type']))

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            img_names = batch['image_filename']

            outputs = model(images, input_ids, attention_mask)
            generated_ids = model.t5.generate(
                inputs_embeds=outputs.final_proj,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

            preds = [model.t5.decode(g, skip_special_tokens=True).lower() for g in generated_ids]
            true_answers = [model.t5.decode(l, skip_special_tokens=True).lower() for l in labels]

            all_predictions.extend(preds)
            all_labels.extend(true_answers)
            all_img_names.extend(img_names)

    bleu_scores = []
    rouge_scores = []

    for pred, ref in zip(all_predictions, all_labels):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5))
        bleu_scores.append(bleu)

        rouge = scorer.score(ref, pred)['rouge1'].fmeasure
        rouge_scores.append(rouge)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    yesno_correct = 0
    yesno_total = 0
    for pred, ref, img_name in zip(all_predictions, all_labels, all_img_names):
        qtype = yesno_dict.get(img_name, 'other')
        if qtype == 'yesno':
            yesno_total += 1
            if pred.strip() == ref.strip():
                yesno_correct += 1
    yesno_accuracy = yesno_correct / yesno_total if yesno_total > 0 else 0.0

    print(f"Avg BLEU: {avg_bleu:.4f}")
    print(f"Avg ROUGE-1: {avg_rouge:.4f}")
    print(f"Yes/No Accuracy: {yesno_accuracy:.4f} ({yesno_correct}/{yesno_total})")

    return all_predictions, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_root = r"C:\Users\Rameshwar\MedCoTVQARad\VQA_RADImageFolder"
    annotation_file = r"C:\Users\Rameshwar\MedCoTVQARad\VQA_RADDatasetPublic.xlsx"
    yesno_csv = r"C:\Users\Rameshwar\MedCoTVQARad\yesno_questions.csv"

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    test_dataset = VQARADDataset(tokenizer, image_root, annotation_file)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = MedCoTModel()
    model.load_state_dict(torch.load("medcot_epoch_final.pth"))
    model.to(device)

    preds, labels = evaluate(model, test_loader, device, yesno_csv)

    for pred, label in zip(preds[:10], labels[:10]):
        print(f"Predicted: {pred} | Ground Truth: {label}")

if __name__ == "__main__":
    main()
