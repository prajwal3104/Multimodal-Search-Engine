import os
import tensorflow as tf
from tensorflow.keras import utils

def download_and_extract_data():
    """Downloads and extracts the COCO dataset if it doesn't exist."""

    root_dir = "datasets"
    annotations_dir = os.path.join(root_dir, "annotations")
    images_dir = os.path.join(root_dir, "train2014")

    # Download caption annotation files
    if not os.path.exists(annotations_dir):
        annotation_zip = utils.get_file(
            "captions.zip",
            cache_dir=os.path.abspath("."),
            origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            extract=True,
        )
        os.remove(annotation_zip)

    # Download image files
    if not os.path.exists(images_dir):
        image_zip = utils.get_file(
            "train2014.zip",
            cache_dir=os.path.abspath("."),
            origin="http://images.cocodataset.org/zips/train2014.zip",
            extract=True,
        )
        os.remove(image_zip)

    print("Dataset is downloaded and extracted successfully.")

if __name__ == "__main__":
    download_and_extract_data()