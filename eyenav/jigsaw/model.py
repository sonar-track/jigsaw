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
    '''
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
    ''' Initialize a simple feed-forward convolutional model
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