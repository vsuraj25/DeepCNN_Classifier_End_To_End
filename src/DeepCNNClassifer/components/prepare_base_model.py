import tensorflow as tf
from DeepCNNClassifer.entity.config_entity import *
from DeepCNNClassifer import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        logger.info(f'Loading VGG16 as base model.')
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )   
        logger.info(f'Saving VGG16(without output layer) as base model at {self.config.base_model}')
        self.save_model(path = self.config.base_model, model = self.model)
        logger.info(f'Base model at {self.config.base_model}')

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            logger.info(f'Freezing the input and hidden layers...')
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            logger.info(f'Freezing till layer - {freeze_till}...')
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        
        logger.info(f'Flattening layers for output...')
        flatten_in = tf.keras.layers.Flatten()(model.output)
        logger.info(f'Building output layer...')
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation = 'softmax'
        )(flatten_in)
        
        logger.info(f'Buiding full model...')
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction
        )

        logger.info(f'Compiling model...')
        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
        )
        logger.info(f'Model built successfully!')
        logger.info(f'Model Summary - {full_model.summary()}')
        
        return full_model

    def update_base_model(self):
        logger.info(f'Updating base model...')
        self.full_model = self._prepare_full_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )
        logger.info(f'Saving model at {self.config.updated_base_model}...')
        self.save_model(path = self.config.updated_base_model, model = self.full_model)
        logger.info(f'Model saved successfully at {self.config.updated_base_model}...')

    @staticmethod
    def save_model(path : Path, model : tf.keras.Model):
        model.save(path)