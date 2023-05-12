import tensorflow as tf
from DeepCNNClassifer.entity.config_entity import ModelEvaluationConfig
from DeepCNNClassifer.utils import *
from DeepCNNClassifer import logger
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
                target_size = self.config.params_img_size[:-1],
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

    @staticmethod
    def load_model(path : Path):
        return tf.keras.models.load_model(path)

    def evaluate(self):
        model = self.load_model(self.config.path)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_scores(self):
        scores = {
            "loss" : self.score[0],
            "accuracy" : self.score[1]
        }
        return save_json(path = Path('scores.json'), data = scores)