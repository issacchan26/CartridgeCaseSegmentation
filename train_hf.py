import json
import torch
import copy
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer

processor = SegformerImageProcessor(size = {"height": 256, "width": 256})

def get_dataset(hf_dataset_identifier):
    ds = load_dataset(hf_dataset_identifier)
    ds = ds.shuffle(seed=1)
    ds = ds["train"].train_test_split(test_size=0.1, seed=8)
    train_ds = ds["train"]
    test_ds = ds["test"]
    filename = "id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    return train_ds, test_ds, id2label, label2id, num_labels

def train_transforms(example_batch):
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 
    images = [jitter(x.convert("RGB")) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=0,
                reduce_labels=processor.do_reduce_labels,
            )
        
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics

def color_map():
    return [[0, 0, 0], [216, 82, 24], [255, 255, 0], [125, 46, 141], [118, 171, 47]]

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(color_map())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img

if __name__ == '__main__':

    hf_dataset_identifier = "issacchan26/gray_bullet"
    pretrained_model_name = "nvidia/mit-b0"
    epochs = 300
    lr = 0.0005
    batch_size = 1
    save_dir = "/path to fine tuned model saving folder"

    # Set transforms
    train_ds, test_ds, id2label, label2id, num_labels = get_dataset(hf_dataset_identifier)
    original_test_ds = copy.deepcopy(test_ds)
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=5,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        logging_steps=1,
        eval_accumulation_steps=10,
        load_best_model_at_end=True,
    )
    metric = evaluate.load("mean_iou")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
