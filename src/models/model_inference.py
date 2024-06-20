import tensorflow as tf
from src.models.model_architectures import create_vision_encoder, create_text_encoder
from src.data.data_utils import create_tfrecord_dataset
from src.utils.config import Config

def generate_image_embeddings(image_paths, model_dir, batch_size):
    """Generates image embeddings."""

    config = Config()
    vision_encoder = create_vision_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=False,
    )

    def read_image(image_path):
        image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        return tf.image.resize(image_array, (299, 299))

    print(f"Generating embeddings for {len(image_paths)} images   ...")
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")
    return image_embeddings

def find_matches(image_embeddings, queries, k=9, normalize=True):
    """Finds relevant images based on text queries."""

    config = Config()
    text_encoder = create_text_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=False,
    )

    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return [[image_paths[idx] for idx in indices] for indices in results]