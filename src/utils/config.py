import os
from environs import Env

env = Env()
env.read_env()

class Config:
    """Configuration settings."""

    root_dir = env("ROOT_DIR", "datasets")
    annotations_dir = os.path.join(root_dir, "annotations")
    images_dir = os.path.join(root_dir, "train2014")
    tfrecords_dir = os.path.join(root_dir, "tfrecords")
    annotation_file = os.path.join(annotations_dir, "captions_train2014.json")
    model_dir = env("MODEL_DIR", "models")
    image_paths = [
        os.path.join(images_dir, f"COCO_train2014_{i:012d}.jpg")
        for i in range(1, 30001)
    ]  # You may need to adjust this based on your dataset

    # Training hyperparameters
    num_projection_layers = env.int("NUM_PROJECTION_LAYERS", 1)
    projection_dims = env.int("PROJECTION_DIMS", 256)
    dropout_rate = env.float("DROPOUT_RATE", 0.1)
    trainable_base_encoders = env.bool("TRAINABLE_BASE_ENCODERS", False)
    learning_rate = env.float("LEARNING_RATE", 0.001)
    weight_decay = env.float("WEIGHT_DECAY", 0.001)
    temperature = env.float("TEMPERATURE", 0.05)
    num_epochs = env.int("NUM_EPOCHS", 5)
    batch_size = env.int("BATCH_SIZE", 256)

    # Inference parameters
    k = env.int("K", 9)  # Number of top results to return