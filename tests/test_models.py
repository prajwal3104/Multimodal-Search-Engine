import unittest
from src.models.model_architectures import create_vision_encoder, create_text_encoder
from src.models.model_training import DualEncoder
from src.utils.config import Config

class TestModels(unittest.TestCase):
    def test_create_vision_encoder(self):
        """Tests the create_vision_encoder function."""

        config = Config()
        vision_encoder = create_vision_encoder(
            num_projection_layers=config.num_projection_layers,
            projection_dims=config.projection_dims,
            dropout_rate=config.dropout_rate,
        )
        self.assertIsInstance(vision_encoder, tf.keras.Model)

    def test_create_text_encoder(self):
        """Tests the create_text_encoder function."""

        config = Config()
        text_encoder = create_text_encoder(
            num_projection_layers=config.num_projection_layers,
            projection_dims=config.projection_dims,
            dropout_rate=config.dropout_rate,
        )
        self.assertIsInstance(text_encoder, tf.keras.Model)

    def test_dual_encoder(self):
        """Tests the DualEncoder class."""

        config = Config()
        vision_encoder = create_vision_encoder(
            num_projection_layers=config.num_projection_layers,
            projection_dims=config.projection_dims,
            dropout_rate=config.dropout_rate,
        )
        text_encoder = create_text_encoder(
            num_projection_layers=config.num_projection_layers,
            projection_dims=config.projection_dims,
            dropout_rate=config.dropout_rate,
        )
        dual_encoder = DualEncoder(text_encoder, vision_encoder, temperature=config.temperature)
        self.assertIsInstance(dual_encoder, tf.keras.Model)

if __name__ == '__main__':
    unittest.main()