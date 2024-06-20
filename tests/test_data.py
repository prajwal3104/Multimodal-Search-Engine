import unittest
from src.data.data_utils import preprocess_captions, write_tfrecords, create_tfrecord_dataset
from src.utils.config import Config

class TestData(unittest.TestCase):
    def test_preprocess_captions(self):
        """Tests the preprocess_captions function."""

        config = Config()
        image_path_to_caption, image_paths = preprocess_captions(config.annotation_file)
        self.assertIsInstance(image_path_to_caption, collections.defaultdict)
        self.assertIsInstance(image_paths, list)

    def test_write_tfrecords(self):
        """Tests the write_tfrecords function."""

        config = Config()
        image_path_to_caption, image_paths = preprocess_captions(config.annotation_file)
        num_examples = write_tfrecords(
            "test.tfrecord", image_paths[:10], image_path_to_caption, 2
        )
        self.assertGreater(num_examples, 0)

    def test_create_tfrecord_dataset(self):
        """Tests the create_tfrecord_dataset function."""

        dataset = create_tfrecord_dataset("test.tfrecord", 1)
        self.assertIsInstance(dataset, tf.data.Dataset)

if __name__ == '__main__':
    unittest.main()