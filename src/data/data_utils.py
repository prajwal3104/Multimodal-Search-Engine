import os
import collections
import json
import numpy as np
import tensorflow as tf

def preprocess_captions(annotations_file):
    """Processes captions from the COCO annotation file."""

    with open(annotations_file, "r") as f:
        annotations = json.load(f)["annotations"]

    image_path_to_caption = collections.defaultdict(list)
    for element in annotations:
        caption = f"{element['caption'].lower().rstrip('.')}"
        image_path = os.path.join("datasets", "train2014", f"COCO_train2014_{element['image_id']:012d}.jpg")
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    print(f"Number of images: {len(image_paths)}")
    return image_path_to_caption, image_paths

def create_tfrecord_example(image_path, caption):
    """Creates a TFRecord example."""

    feature = {
        "caption": tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption.encode()])),
        "raw_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.read_file(image_path).numpy()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(file_name, image_paths, image_path_to_caption, captions_per_image):
    """Writes data to TFRecord files."""

    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_tfrecord_example(image_path_list[example_idx], caption_list[example_idx])
            writer.write(example.SerializeToString())

    return example_idx + 1

def create_tfrecord_dataset(file_pattern, batch_size):
    """Creates a TFRecord dataset."""

    feature_description = {
        "caption": tf.io.FixedLenFeature([], tf.string),
        "raw_image": tf.io.FixedLenFeature([], tf.string),
    }

    def read_example(example):
        features = tf.io.parse_single_example(example, feature_description)
        raw_image = features.pop("raw_image")
        features["image"] = tf.image.resize(
            tf.image.decode_jpeg(raw_image, channels=3), size=(299, 299)
        )
        return features

    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .batch(batch_size)
    )