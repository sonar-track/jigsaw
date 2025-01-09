''' Author: Phong D. Vo, phong.vodinh@gmail.com

    Model implementation from the paper:
    https://openaccess.thecvf.com/content/CVPR2022W/LXCV/html/Preciado-Grijalva_Self-Supervised_Learning_for_Sonar_Image_Classification_CVPRW_2022_paper.html
'''
from tensorflow import keras


def create_feature_extractor(
        image_width: int=32,
        image_height: int=32,
        num_channels: int=1,
        dropout: float=0.3,
        alpha: float=0.75,
        minimalistic: bool=True,
        include_top: bool=False,
        include_preprocessing: bool=False,
        weights: str=None,
        preset_model: str='default',
        weights_filepath: str=None
    ):
    ''' Initialize a simple feed-forward convolutional model

    This function creates a feature extractor model that can be used for sonar image classification tasks. 
    It supports two preset models: 'default' and 'mobilenetv3small'. 
    The 'default' model is a simple feed-forward convolutional model, while 'mobilenetv3small' is a pre-trained MobileNetV3Small model. 
    The model can be customized by specifying the image dimensions, number of channels, dropout rate, and other parameters.

    Parameters:
    - image_width (int): The width of the input images.
    - image_height (int): The height of the input images.
    - num_channels (int): The number of channels in the input images.
    - dropout (float): The dropout rate for the model.
    - alpha (float): The alpha parameter for the MobileNetV3Small model.
    - minimalistic (bool): A flag indicating whether to use the minimalistic version of the MobileNetV3Small model.
    - include_top (bool): A flag indicating whether to include the classification head in the model.
    - include_preprocessing (bool): A flag indicating whether to include preprocessing layers in the model.
    - weights (str): The path to the pre-trained weights file.
    - preset_model (str): The type of preset model to use. Supported values are 'default' and 'mobilenetv3small'.
    - weights_filepath (str): The path to the custom weights file to load.

    Returns:
    - keras.models.Model: The created feature extractor model.
    '''
    inp = keras.Input(shape=(image_height, image_width, num_channels))

    if preset_model == 'default':

        t = keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(inp)  # 96
        t = keras.layers.BatchNormalization()(t)
        t = keras.layers.MaxPooling2D((2, 2), padding='same')(t)
        t = keras.layers.Dropout(0.3)(t)
        
        t = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(t) # 256
        t = keras.layers.BatchNormalization()(t)
        t = keras.layers.MaxPooling2D((2, 2), padding='same')(t)
        t = keras.layers.Dropout(0.3)(t)

        t = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(t) # 384
        t = keras.layers.BatchNormalization()(t)
        t = keras.layers.MaxPooling2D((2,2), padding='same')(t)
        t = keras.layers.Dropout(dropout)(t)
        
        t = keras.layers.Flatten()(t)
        
        feature_extractor = keras.models.Model(inputs=inp, outputs=t, name='feature_extractor')
    
    elif preset_model == 'mobilenetv3small':

        feature_extractor = keras.applications.MobileNetV3Small(
            input_tensor=inp,
            alpha=alpha,
            minimalistic=minimalistic,
            include_top=include_top,
            include_preprocessing=include_preprocessing,
            weights=weights,
            dropout_rate=dropout
        )
    else:
        raise NotImplementedError

    if weights_filepath:
        print(f'Reuse weights from {weights_filepath}')
        feature_extractor.load_weights(weights_filepath)

    return feature_extractor


def create_jigsaw_classifier(
        feature_extractor: keras.models.Model,
        ncells_x: int=3, 
        ncells_y: int=3,
        num_classes: int=1,
        dropout: float=0.3,
    ):
    ''' Creates a jigsaw puzzle classifier model based on a given feature extractor.

    This function constructs a model that takes a jigsaw puzzle as input, applies a feature extractor to each piece, 
    and then classifies the puzzle. The model consists of a time-distributed application of the feature extractor, 
    followed by a series of dense layers for classification.

    Parameters:
    - feature_extractor: keras.models.Model, the feature extractor model to be applied to each piece of the jigsaw puzzle.
    - ncells_x: int, the number of cells in the x-direction of the jigsaw puzzle.
    - ncells_y: int, the number of cells in the y-direction of the jigsaw puzzle.
    - num_classes: int, the number of classes for classification.
    - dropout: float, the dropout rate for the dense layers.

    Returns:
    - keras.models.Model, the constructed jigsaw puzzle classifier model.
    '''
    input_shape = (ncells_x*ncells_y, *feature_extractor.input.shape[1:])
    inp = keras.Input(shape=input_shape)

    t = keras.layers.TimeDistributed(feature_extractor)(inp)
    t = keras.layers.Flatten()(t)
    t = keras.layers.Dense(64, activation='relu')(t)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.Dense(32, activation='relu')(t)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.Dropout(dropout)(t)
    t = keras.layers.Dense(num_classes, activation='softmax')(t)

    model = keras.models.Model(inputs=inp, outputs=t, name='puzzle_solver')

    return model


def create_pyramid_feature_extractor(
        image_width: int=32,
        image_height: int=32,
        num_channels: int=1,
        dropout: float=0.3,
        alpha: float=0.75,
        minimalistic: bool=True,
        include_top: bool=False,
        include_preprocessing: bool=False,
        weights: str=None,
        preset_model: str='default',
        weights_filepath: str=None
    ):
    ''' Creates a pyramid feature extractor model based on a given preset model.

    This function constructs a pyramid feature extractor model based on a specified preset model. The model consists of a base model, 
    followed by additional layers to extract features at different scales, forming a pyramid structure. The pyramid layers are defined 
    by their layer names within the base model.

    Parameters:
    - image_width: int, the width of the input image.
    - image_height: int, the height of the input image.
    - num_channels: int, the number of channels in the input image.
    - dropout: float, the dropout rate for the base model.
    - alpha: float, the alpha parameter for the base model (MobileNetV3Small).
    - minimalistic: bool, a flag indicating whether to use a minimalistic model (MobileNetV3Small).
    - include_top: bool, a flag indicating whether to include the top layer in the base model.
    - include_preprocessing: bool, a flag indicating whether to include preprocessing layers in the base model.
    - weights: str, the path to the weights file for the base model.
    - preset_model: str, the name of the preset model to use.
    - weights_filepath: str, the path to the weights file for the entire model.

    Returns:
    - keras.models.Model, the constructed pyramid feature extractor model.
    '''
    inp = keras.Input(shape=(image_height, image_width, num_channels))

    if preset_model == 'mobilenetv3small':

        feature_extractor = keras.applications.MobileNetV3Small(
            input_tensor=inp,
            alpha=alpha,
            minimalistic=minimalistic,
            include_top=include_top,
            include_preprocessing=include_preprocessing,
            weights=weights,
            dropout_rate=dropout
        )
    else:
        raise NotImplementedError

    pyramid_layers = ['re_lu_16', 're_lu_6', 're_lu_2']
    pyramid_outputs = [
        feature_extractor.get_layer(layer_name).output for layer_name in pyramid_layers
    ]

    if weights_filepath:
        print(f'Reuse weights from {weights_filepath}')
        feature_extractor.load_weights(weights_filepath)

    pyramid_extractor = keras.Model(
        inputs=inp,
        outputs=[feature_extractor.output,] + pyramid_outputs
    )

    return pyramid_extractor