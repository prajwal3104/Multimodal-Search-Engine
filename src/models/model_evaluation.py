import tensorflow as tf
from src.models.model_architectures import create_vision_encoder, create_text_encoder
from src.data.data_utils import create_tfrecord_dataset
from src.utils.config import Config

def evaluate_model(valid_dataset, model_dir):
    """Evaluates the trained model."""

    config = Config()
    vision_encoder = create_vision_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=False,
    )
    text_encoder = create_text_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=False,
    )
    dual_encoder = tf.keras.models.load_model(os.path.join(model_dir, "dual_encoder"))

    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Number of examples (caption-image pairs): {len(valid_dataset)}")

    # Perform evaluation
    evaluation_results = dual_encoder.evaluate(valid_dataset, verbose=1)

    print(f"Evaluation results: {evaluation_results}")