import json
import torch
import evaluate
import random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, default_data_collator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import (ColorJitter, Compose, RandomRotation, Normalize, RandomHorizontalFlip, functional)

processor = SegformerImageProcessor(size = {"height": 256, "width": 256})

class Params(object):
    def __init__(self, hf_dataset_identifier, pretrained_model_name, epochs, lr, batch_size, checkpoints_path):
        self.hf_dataset_identifier = hf_dataset_identifier
        self.pretrained_model_name = pretrained_model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.checkpoints_path = checkpoints_path

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

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        original_width, original_height = img.size
        pad_height = size - original_height if original_height < size else 0
        pad_width = size - original_width if original_width < size else 0
        img = functional.pad(img, (0, 0, pad_width, pad_height), fill=fill)
    return img

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = functional.resize(image, self.size)
        target = functional.resize(target, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return image, target

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        image = functional.rotate(image, self.degrees)
        target = functional.rotate(target, self.degrees)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return image, target

class PILToTensor:
    def __call__(self, image, target):
        image = functional.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = functional.convert_image_dtype(image, self.dtype)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image, target

train_transforms = Compose([
    Resize(size=(256,256)),
    RandomRotation(degrees=180),
    RandomHorizontalFlip(flip_prob=0.5),
    PILToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

val_transforms = Compose([
    Resize(size=(256,256)),
    PILToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

def preprocess_train(example_batch):
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 
    pixel_values = []
    labels = []
    for image, target in zip(example_batch["pixel_values"], example_batch["label"]):
        image, target = train_transforms(jitter(image.convert("RGB")), target)
        pixel_values.append(image)
        labels.append(target)
    encoding = {}
    encoding["pixel_values"] = torch.stack(pixel_values)
    encoding["labels"] = torch.stack(labels)
    return encoding

def preprocess_val(example_batch):
    pixel_values = []
    labels = []
    for image, target in zip(example_batch["pixel_values"], example_batch["label"]):
        image, target = val_transforms(image.convert("RGB"), target)
        pixel_values.append(image)
        labels.append(target)
    encoding = {}
    encoding["pixel_values"] = torch.stack(pixel_values)
    encoding["labels"] = torch.stack(labels)
    return encoding

def get_transform_dataloader(train_ds, test_ds):
    train_dataset = train_ds.with_transform(preprocess_train)
    eval_dataset = test_ds.with_transform(preprocess_val)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)
    return train_dataloader, eval_dataloader

def train(train_loader):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float().item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss

def validation(eval_dataloader):
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False
        )
        predictions = upsampled_logits.argmax(dim=1)
        references = batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metrics = metric.compute(
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=False
    )
    return eval_metrics

if __name__ == '__main__':

    args = Params(
        hf_dataset_identifier = "issacchan26/gray_bullet",
        pretrained_model_name = '/path to pretrained model folder from Hugging Face', # path to Hugging Face pretrained model
        epochs = 100,
        lr = 0.0005,
        batch_size = 1,
        checkpoints_path = "/path to/checkpoints/"  # path to checkpoints folder
        )

    train_ds, test_ds, id2label, label2id, num_labels = get_dataset(args.hf_dataset_identifier)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name, id2label=id2label, label2id=label2id, num_labels=num_labels)
    model.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    scheduler = LinearLR(optimizer)
    metric = evaluate.load("mean_iou")
    
    best_mean_iou = 0
    best_mean_accuracy = 0
    best_epoch = 0

    for epoch in tqdm(range(1, args.epochs+1)):
        train_dataloader, eval_dataloader = get_transform_dataloader(train_ds, test_ds)
        train_loss = train(train_dataloader)
        model.save_pretrained(args.checkpoints_path + 'latest')
        if epoch%2 == 0:
            eval_metrics = validation(eval_dataloader)
            mean_iou = eval_metrics['mean_iou']
            mean_accuracy = eval_metrics['mean_accuracy']
            if (mean_iou > best_mean_iou) and (mean_accuracy > best_mean_accuracy):
                best_mean_iou = mean_iou
                best_mean_accuracy = mean_accuracy
                best_epoch = epoch
                model.save_pretrained(args.checkpoints_path + 'best')
            print(f'Best mean IOU: {best_mean_iou:.4f}, Best Epoch: {best_epoch}')
