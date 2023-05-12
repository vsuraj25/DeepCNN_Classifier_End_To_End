import tensorflow as tf
from DeepCNNClassifer.entity.config_entity import ModelTrainingConfig
from DeepCNNClassifer import logger
from pathlib import Path

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
    
    def get_base_model(self):
        logger.info(f"Getting Base Model....")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        
        logger.info(f"Rescaling and generating validation split....")
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
        )

        dataflow_kwargs = dict(
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = "bilinear"
            )
        
        valid_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_data_generator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle = False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmented:
            logger.info(f"Performing Data Augumentation....")
            training_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range= 40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                ** datagenerator_kwargs
            )

        else:
            training_datagenerator= valid_data_generator

        self.train_generator = training_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset = "training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path : Path, model = tf.keras.Model):
        model.save(path)
        logger.info(f"Model Saved at path - {path}")

    def train(self, callback_list):
        self.steps_for_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epoch,
            steps_per_epoch = self.steps_for_epoch,
            validation_steps= self.validation_steps,
            validation_data = self.valid_generator,
            callbacks = callback_list
        )
        logger.info(f"Training the model for {self.config.params_epoch} epochs....")

        self.save_model(
            path = self.config.trained_model_path, 
            model = self.model
        )
    