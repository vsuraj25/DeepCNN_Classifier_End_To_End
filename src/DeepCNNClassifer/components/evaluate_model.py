import tensorflow as tf
from DeepCNNClassifer.entity.config_entity import ModelEvaluationConfig
from DeepCNNClassifer.utils import *
from DeepCNNClassifer.sec import *
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

    def _valid_generator(self):
        
        logger.info(f"Rescaling and generating test split....")
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

        logger.info(f"Generating test data using flow from directory...")
        self.valid_generator = valid_data_generator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            shuffle = False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path : Path):
        logger.info(f"Loading the model from {path}...")
        return tf.keras.models.load_model(path)

    def evaluate(self):
        logger.info(f"Evaluating the model...")
        self.model = self.load_model(self.config.path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_scores(self):
        scores = {
            "loss" : self.score[0],
            "accuracy" : self.score[1]
        }
        logger.info(f"Saving the scores as json at 'scores.json'...")
        return save_json(path = Path('scores.json'), data = scores)
    
    def log_into_mlflow(self):
        logger.info(f"MLFlow tracking started...")
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            logger.info(f"Logging all parameters in mlflow...")
            mlflow.log_params(self.config.all_params)
            logger.info(f"Logging all metrics in mlflow...")
            mlflow.log_metrics({
                "loss" : self.score[0],
                "accuracy" : self.score[1]
            })
            if tracking_uri_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name = "VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")

            logger.info(f"MLFlow tracking complete, you can visualize your model results using the command `mlflow ui` or globally at https://dagshub.com/vsuraj25/DeepCNN_Classifier_End_To_End.mlflow")