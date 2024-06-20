import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from src.models.model_architectures import create_vision_encoder, create_text_encoder
from src.utils.config import Config

class DualEncoder(tf.keras.Model):
    """The Dual Encoder model."""

    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        """Forward pass."""

        with tf.device("/gpu:0"):
            caption_embeddings = self.text_encoder(features["caption"], training=training)
        with tf.device("/gpu:1"):
            image_embeddings = self.image_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        """Calculates the loss."""

        logits = (
            tf.matmul(caption_embeddings, image_embeddings, transpose_b=True)
            / self.temperature
        )
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings           , transpose_b=True
        )
        captions_similarity = tf.matmul(
            caption_embeddings, caption_embeddings, transpose_b=True
        )
        targets = tf.keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        captions_loss = tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        images_loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(           targets), y_pred=tf.transpose(logits), from_logits=True
        )
        return (captions_loss + images_loss) / 2

    def train_step(self, features):
        """Training step."""

        with tf.GradientTape() as tape:
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        """Evaluation step."""

        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

def train_dual_encoder(train_dataset, valid_dataset, num_epochs, batch_size):
    """Trains the dual encoder model."""

    config = Config()
    vision_encoder = create_vision_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=config.trainable_base_encoders,
    )
    text_encoder = create_text_encoder(
        num_projection_layers=config.num_projection_layers,
        projection_dims=config.projection_dims,
        dropout_rate=config.dropout_rate,
        trainable=config.trainable_base_encoders,
    )
    dual_encoder = DualEncoder(
        text_encoder, vision_encoder, temperature=config.temperature
    )
    dual_encoder.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=config.learning_rate, weight_decay=config.weight_decay
        )
    )

    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Number of examples (caption-image pairs): {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {int(np.ceil(len(train_dataset) / batch_size))}")

    # Create a learning rate scheduler callback.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = dual_encoder.fit(
        train_dataset,
        epochs=       num_epochs,
        validation_data=valid_dataset,
        callbacks=[reduce_lr, early_stopping],
    )
    print("Training completed. Saving vision and text encoders...")
    vision_encoder.save(os.path.join(config.model_dir, "vision_encoder"))
    text_encoder.save(os.path.join(config.model_dir, "text_encoder"))
    print("Models are saved.")
    return dual_encoder, history