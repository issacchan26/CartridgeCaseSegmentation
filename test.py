import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision.transforms import functional

class Params(object):
    def __init__(self, hf_dataset_identifier, pretrained_model_name, prediction_save_path):
        self.hf_dataset_identifier = hf_dataset_identifier
        self.pretrained_model_name = pretrained_model_name
        self.prediction_save_path = prediction_save_path

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

def color_map():
    return [
        [0, 0, 0],
        [133, 13, 13],
        [85, 126, 85],
        [111, 77, 118],
        [78, 177, 177],
    ]

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(color_map())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img

def get_prediction(original_test_ds, model_name, prediction_save_path):
    with torch.no_grad():
        processor = SegformerImageProcessor(size = {"height": 256, "width": 256})
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.eval()
        
        for image_idx in range(len(original_test_ds)):
            image = original_test_ds[image_idx]['pixel_values'].convert("RGB")
            gt_seg = original_test_ds[image_idx]['label']
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

            # First, rescale logits to original image size
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (height, width)
                mode='bilinear',
                align_corners=False
            )

            # Second, apply argmax on the class dimension
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_img = get_seg_overlay(image, pred_seg)
            gt_img = get_seg_overlay(image, np.array(gt_seg))
            f, axs = plt.subplots(1, 2)
            f.set_figheight(30)
            f.set_figwidth(50)
            axs[0].set_title("Prediction", {'fontsize': 40})
            axs[0].imshow(pred_img)
            axs[1].set_title("Ground truth", {'fontsize': 40})
            axs[1].imshow(gt_img)
            plt.savefig(prediction_save_path + f'plt{image_idx}.png')
    return

if __name__ == '__main__':

    args = Params(
        hf_dataset_identifier = "issacchan26/gray_bullet",
        pretrained_model_name = '/path to/checkpoints/best',  # path to model folder
        prediction_save_path = '/path to/prediction/', # path to saving folder
        )

    train_ds, original_test_ds, id2label, label2id, num_labels = get_dataset(args.hf_dataset_identifier)
    get_prediction(original_test_ds, args.pretrained_model_name, args.prediction_save_path)
