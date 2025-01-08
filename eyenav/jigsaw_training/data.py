'''
'''
import itertools, random, typing
import tensorflow as tf
from tensorflow import keras 
from functools import partial


def randomize_permutation_patterns(
        num_puzzles: int, 
        num_patterns: int=None
    ) -> typing.List[typing.List[int]]:
    ''' Randomly create a set of non-duplicate permutation patterns from 0..num_puzzles

    Args:
        num_patterns:   If set to None then all possible permutations are returned.
                        Otherwise randomly select a subset of `num_patterns`
    '''
    all_perm_patterns = list(set(itertools.permutations(list(range(num_puzzles)))))
    if num_patterns is None:
        return all_perm_patterns
    if num_patterns > len(all_perm_patterns):
        raise ValueError(f'{num_patterns} is larger than all possible patterns')
    patterns = random.sample(all_perm_patterns, num_patterns)
    return patterns


@tf.function
def random_crop_and_resize_patch_from_image(
        image: tf.Tensor,
        patch_width: int,
        patch_height: int,
        crop_range: typing.List[float]=[0.8, 4.0],
        contrast_range: typing.List[float]=[0.8, 1.2],
        brightness_delta: float=0.2,
    ) -> tf.Tensor:
    ''' Randomly pick patch center wrt pdf of X and Y axes such that cropped patch
        always is unlikely to be an empty patch, but contains textures of different sorts.

        Return:
         A single patch of tf.Tensor type
    '''
    crop_height = tf.random.uniform(shape=(), minval=int(patch_height * crop_range[0]), maxval=int(patch_height * crop_range[1]), dtype=tf.int32)
    crop_width = tf.random.uniform(shape=(), minval=int(patch_width * crop_range[0]), maxval=int(patch_width * crop_range[1]), dtype=tf.int32)
    half_height = crop_height//2
    half_width = crop_width//2
    image = tf.cond(tf.equal(tf.rank(image), 2),
                    lambda: tf.expand_dims(image, axis=-1),
                    lambda: image)
    prob_x = tf.reduce_sum(image, axis=[0, 2])[half_width:-half_width]
    prob_x = prob_x / tf.reduce_sum(prob_x)

    prob_y = tf.reduce_sum(image, axis=[1, 2])[half_height:-half_height]
    prob_y = prob_y / tf.reduce_sum(prob_y)

    offset_width = tf.random.categorical([tf.math.log(prob_x)], 1, dtype=tf.int32)
    offset_height = tf.random.categorical([tf.math.log(prob_y)], 1, dtype=tf.int32)

    offset_height = tf.squeeze(offset_height)
    offset_width = tf.squeeze(offset_width)
    patch = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    patch = tf.image.resize(patch, (patch_height, patch_width))
    patch = tf.image.convert_image_dtype(patch, tf.float32)

    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    patch = tf.image.random_contrast(patch, contrast_range[0], contrast_range[1])
    patch = tf.image.random_brightness(patch, max_delta=brightness_delta)
    patch = tf.squeeze(patch)
    return patch


@tf.function
def scramble_patch_to_puzzle(
        patch: tf.Tensor,
        puzzle_patterns: typing.List[typing.List[int]],
        ncells_x: int=3,
        ncells_y: int=3,
    ) -> tf.Tensor:
    ''' From a given patch, partitioning it to ncells_x * ncells_y blocks and shuffle
        the blocks wrt puzzle_pattern, then return the puzzled patch
     '''
    puzzle_patterns = tf.convert_to_tensor(puzzle_patterns)
    patch = tf.cond(tf.equal(tf.rank(patch), 2), 
                    lambda: tf.expand_dims(patch, axis=-1), 
                    lambda: patch)
    patch_height = tf.shape(patch)[0]
    patch_width = tf.shape(patch)[1]
    channels = tf.shape(patch)[2]
    blocks = tf.reshape(patch, (ncells_y, patch_height//ncells_y, ncells_x, patch_width//ncells_x, channels))
    blocks = tf.transpose(blocks, [0, 2, 1, 3, 4])
    blocks = tf.reshape(blocks, (ncells_x*ncells_y, patch_height//ncells_y, patch_width//ncells_x, channels))

    num_classes = tf.shape(puzzle_patterns)[0]
    uniform_probs = tf.ones((1, len(puzzle_patterns)), dtype=tf.float32) / tf.cast(len(puzzle_patterns), tf.float32)
    selected = tf.random.categorical(tf.math.log(uniform_probs), 1, dtype=tf.int32)
    label = selected[0][0]
    pattern = puzzle_patterns[label]

    puzzled = tf.gather(blocks, pattern, axis=0, batch_dims=None)
    #puzzled = tf.squeeze(puzzled)
    one_hot_label = tf.one_hot(label, num_classes)
    return puzzled, one_hot_label


def build_data_pipeline(
        sonar_images_directory: str,
        puzzle_patterns: typing.List[typing.List[int]],
        ncells_x: int=3,
        ncells_y: int=3,
        train_val_split: float=None,
        color_mode: str='grayscale',
        batch_size: int=32,
        image_width: int=1024,
        image_height: int=1536,
        patch_width: int=96,
        crop_range: typing.List[float]=[0.8, 4.0],
        contrast_range: typing.List[float]=[0.8, 1.2],
        brightness_delta: float=0.3,
        epochs: int=1,
        patch_height: int=96,
        shuffle: bool=True,
        seed: int=None,
        subset: str=None,
    ) -> tf.data.Dataset:
    ''' Create a iterable dataset of sonar image patches
        randomly cropped from (randomly) picked sonar snapshots 
        of a given dataset directory
    '''
    dataset = keras.utils.image_dataset_from_directory(
        sonar_images_directory,
        validation_split=train_val_split,
        color_mode=color_mode,
        batch_size=1,
        image_size=(image_height, image_width),
        crop_to_aspect_ratio=True,
        shuffle=shuffle,
        seed=seed,
        subset=subset,
        label_mode=None
    )

    dataset = dataset.repeat(epochs)
    dataset = dataset.map(lambda image: tf.squeeze(image,axis=0))
    dataset = dataset.map(
        partial(random_crop_and_resize_patch_from_image,
                patch_height=patch_height,
                patch_width=patch_width,
                crop_range=crop_range,
                contrast_range=contrast_range,
                brightness_delta=brightness_delta))
    dataset = dataset.map(
        partial(scramble_patch_to_puzzle,
                puzzle_patterns=puzzle_patterns,
                ncells_x=ncells_x,
                ncells_y=ncells_y))
    dataset = dataset.batch(
        batch_size=batch_size,
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset