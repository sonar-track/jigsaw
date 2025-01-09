import unittest
import numpy as np
import tensorflow as tf


class TestData(unittest.TestCase):

    def test_randomize_patterns(self):

        from sonar.jigsaw_training.data import randomize_permutation_patterns

        patterns = randomize_permutation_patterns(4)
        self.assertTrue((3, 1, 2, 0) in patterns)

        patterns = randomize_permutation_patterns(2)
        self.assertTrue(set([(0,1), (1, 0)]) == set(patterns))


    def test_random_crop(self):
        
        from sonar.jigsaw_training.data import random_crop_patch_from_image

        random_image = np.random.rand(1024, 512)
        random_image = tf.convert_to_tensor(random_image, tf.float32)
        patch_size = 96
        patch = random_crop_patch_from_image(random_image, patch_size, patch_size)
        self.assertEqual(patch.shape, (patch_size, patch_size))

    def test_puzzle(self):

        from sonar.jigsaw_training.data import scramble_patch_to_puzzle, randomize_permutation_patterns

        patch = tf.random.uniform((96, 96, 3), dtype=tf.float32)
        num_patterns = 10
        patterns = randomize_permutation_patterns(9, num_patterns)
        puzzle, label = scramble_patch_to_puzzle(patch, patterns, ncells_x=3, ncells_y=3)
        self.assertEqual(puzzle.shape, (9, 32, 32, 3))        
        self.assertEqual(label.shape, (num_patterns))

    def test_dataset(self):

        image_dir = '../datasets/uatd/train'

        from sonar.jigsaw_training.data import build_data_pipeline, randomize_permutation_patterns

        num_patterns = 10
        puzzle_patterns = randomize_permutation_patterns(9, num_patterns)

        dataset = build_data_pipeline(
            image_dir,
            puzzle_patterns,
            ncells_x=3,
            ncells_y=3,
            batch_size=2,
            patch_width=96,
            patch_height=96
        )

        dataiter = dataset.as_numpy_iterator()
        batch_image, batch_label = next(dataiter)
        self.assertEqual(batch_image.shape, (2, 9, 32, 32, 1))
        self.assertEqual(batch_label.shape, (2, num_patterns))

if __name__ == '__main__':
    unittest.main() 
