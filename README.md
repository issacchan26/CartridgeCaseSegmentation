# Cartridge Case Images Segmentation
This repo provides segmentation model for Cartridge Case part segmentation tasks with Segformer, it will segment the images into {"0": "unlabeled", "1": "breech-face", "2": "aperture-shear", "3": "firing-pin-impression", "4": "firing-pin-drag", "5": "other"}. The trained model is provided in [checkpoints folder](checkpoints).
## Getting Started
This repo is tested with Conda environment and Python 3.9 under Linux os, please run below command to install dependencies
```
pip install -r requirements.txt
```

## Data Annotation
[Segments.ai](https://segments.ai/) is used to annotate the images, before annotating the images, the data are converted into grayscale to simulate 3D images with [convert_color.py](convert_color.py)

## Transfer Learning
Firstly we used [train_hf.py](train_hf.py) for fine-tuning the model and save the fine-tuned model in local. The pretrained model we used is nvidia/mit-b0. Then we use [train.py](train.py) to train the fine-tuned model with image augmentations to further improve model performance.  

## Model Training
Please use [train.py](train.py) to train the model, modify the below arguments before training
```
args = Params(
        hf_dataset_identifier = "issacchan26/gray_bullet", 
        pretrained_model_name = '/path to pretrained model folder from Hugging Face', # path to pretrained model
        epochs = 100,
        lr = 0.0005,
        batch_size = 1,
        checkpoints_path = "/path to/checkpoints/"  # path to checkpoints saving folder
        )
```
## Inference
Please use [inference.py](inference.py) to infer the images, put all the images you would like to infer inside a folder (below we use 'infer_query' as folder name). Modify below path before running:
```
pretrained_model_name = '/path to/checkpoints/best',  # path to model folder
prediction_save_path = '/path to/prediction/', # path to saving folder
infer_folder = '/path to/infer_query/' # path to the image folder to be inferred
```

## To reproduce the validation results
1. [test.py](test.py)  
  It is used to reproduce the validation results of our fine-tuned model  
2. [infer_hf_ds.py](infer_hf_ds.py)  
  It is used to infer the dataset from Hugging Face  

Please modify the below path before running  
```
pretrained_model_name = '/path to/checkpoints/best',  # path to model folder
prediction_save_path = '/path to/prediction/', # path to saving folder
```
