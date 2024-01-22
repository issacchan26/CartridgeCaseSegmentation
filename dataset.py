from segments import SegmentsClient
from segments.huggingface import release2dataset
from segments.utils import get_semantic_bitmap

# Get a specific dataset release
client = SegmentsClient("57a9104c1f2ade4f5e8f26af4c58c6cf453e5810")
release = client.get_release("issacchan26/gray_bullet", "v0.1")
hf_dataset = release2dataset(release)

def convert_segmentation_bitmap(example):
    return {
        "label.segmentation_bitmap":
            get_semantic_bitmap(
                example["label.segmentation_bitmap"],
                example["label.annotations"],
                id_increment=0,
            )
    }

semantic_dataset = hf_dataset.map(
    convert_segmentation_bitmap,
)

semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'label')
semantic_dataset = semantic_dataset.remove_columns(['name', 'uuid', 'status', 'label.annotations'])

semantic_dataset.push_to_hub("issacchan26/gray_bullet") # This is the name of a HF user/dataset

