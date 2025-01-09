import os, keras, typing

from sonar.jigsaw.model import create_jigsaw_classifier
from sonar.jigsaw.model import create_feature_extractor
from sonar.jigsaw_training.data import build_data_pipeline
from sonar.jigsaw_training.data import randomize_permutation_patterns
from sonar.jigsaw_training.utils import logger


def train(
        train_image_dirpath: str='auto',
        val_image_dirpath: str='auto',
        log_dirpath: str='auto',
        checkpoint_dirpath: str='auto',
        trained_classifier_filepath: str='auto',
        trained_backbone_filepath: str='auto',
        image_width: int=512,
        image_height: int=1024,
        num_classes: int=10,
        num_cells_x: int=3,
        num_cells_y: int=3,
        dropout: float=0.1,
        batch_size: int=64,
        patch_height: int=32,
        patch_width: int=32,
        crop_range: typing.List[float]=[0.8, 4.0],
        contrast_range: typing.List[float]=[0.8, 1.2],
        brightness_delta: float=0.3,
        preset_model: str='default',
        shuffle: bool=True,
        epochs: int=1,
        initial_epoch: int=0,
        seed: int=9999,
        patience: int=10,
        optimizer: str='adam',
    ):
    '''
    Trains a jigsaw puzzle classifier model on sonar image datasets.

    This function orchestrates the training process of a jigsaw puzzle classifier model on sonar image datasets. 
    It creates a feature extractor model, a jigsaw puzzle classifier model, and trains the classifier model on the provided training dataset. 
    The training process involves data preprocessing, model compilation, and model fitting.

    Parameters:
    - train_image_dirpath (str): The directory path to the training sonar images.
    - val_image_dirpath (str): The directory path to the validation sonar images.
    - log_dirpath (str): The directory path to save training logs.
    - checkpoint_dirpath (str): The directory path to save model checkpoints.
    - trained_classifier_filepath (str): The file path to save the trained classifier model.
    - trained_backbone_filepath (str): The file path to save the trained backbone model.
    - image_width (int): The width of the input sonar images.
    - image_height (int): The height of the input sonar images.
    - num_classes (int): The number of classes for classification.
    - num_cells_x (int): The number of cells in the x-direction of the jigsaw puzzle.
    - num_cells_y (int): The number of cells in the y-direction of the jigsaw puzzle.
    - dropout (float): The dropout rate for the classifier model.
    - batch_size (int): The batch size for training.
    - patch_height (int): The height of the patch.
    - patch_width (int): The width of the patch.
    - crop_range (typing.List[float]): The range of crop sizes for data augmentation.
    - contrast_range (typing.List[float]): The range of contrast values for data augmentation.
    - brightness_delta (float): The maximum brightness delta for data augmentation.
    - preset_model (str): The type of preset model to use for the feature extractor.
    - shuffle (bool): A flag indicating whether to shuffle the training dataset.
    - epochs (int): The number of epochs for training.
    - initial_epoch (int): The initial epoch for training.
    - seed (int): The seed for random operations.
    - patience (int): The patience for early stopping.
    - optimizer (str): The optimizer to use for training.

    Returns:
    - None: This function does not return any value. It trains the model and saves it to the specified file paths.
    '''
    backbone = create_feature_extractor(
        image_width=patch_width//num_cells_y,
        image_height=patch_height//num_cells_x,
        num_channels=1,
        preset_model=preset_model)

    print(backbone.summary())

    classifier = create_jigsaw_classifier(
        feature_extractor=backbone,
        ncells_x=num_cells_x,
        ncells_y=num_cells_y,
        num_classes=num_classes,
        dropout=dropout)

    print(classifier.summary())

    keras.utils.plot_model(classifier, 'model.png', show_shapes=True)

    num_puzzles = num_cells_x * num_cells_y
    puzzle_patterns = randomize_permutation_patterns(
        num_puzzles,
        num_patterns=num_classes
    )

    train_dataset = build_data_pipeline(
        train_image_dirpath,
        puzzle_patterns=puzzle_patterns,
        ncells_x=num_cells_x,
        ncells_y=num_cells_y,
        batch_size=batch_size,
        image_width=image_width,
        image_height=image_height,
        patch_height=patch_height,
        patch_width=patch_width,
        crop_range=crop_range,
        contrast_range=contrast_range,
        brightness_delta=brightness_delta,
        shuffle=shuffle,
        epochs=epochs,
        seed=seed)
    
    val_dataset = build_data_pipeline(
        val_image_dirpath,
        puzzle_patterns=puzzle_patterns,
        ncells_x=num_cells_x,
        ncells_y=num_cells_y,
        batch_size=batch_size,
        image_width=image_width,
        image_height=image_height,
        patch_height=patch_height,
        patch_width=patch_width,
        seed=seed)

    filepath = os.path.join(checkpoint_dirpath, 'jigsaw_{epoch}_{val_loss}')
    classifier.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics='accuracy',
    )

    steps_per_epoch = train_dataset.cardinality().numpy() // (epochs - initial_epoch)
    validation_steps = val_dataset.cardinality().numpy()
    
    logger.info(f'Step per epoch is estimated as {steps_per_epoch}')
    logger.info(f'Validation steps is estimated as {validation_steps}')

    classifier.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch, 
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath, save_best_only=True),
            keras.callbacks.EarlyStopping(patience=patience, verbose=1, restore_best_weights=True),
            keras.callbacks.CSVLogger(os.path.join(log_dirpath, 'training.log'), append=True)
        ]
    )
    backbone.save(trained_backbone_filepath)
    classifier.save(trained_classifier_filepath)
    logger.info(f'Final feature extractor saved at {trained_backbone_filepath}')
    logger.info(f'Final puzzle classifier saved at {trained_classifier_filepath}')