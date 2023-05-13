from DeepCNNClassifer.utils import *
from DeepCNNClassifer.sec import *
from DeepCNNClassifer.constants import *
from DeepCNNClassifer.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, 
                                                   PrepareCallbackConfig, ModelTrainingConfig,
                                                   ModelEvaluationConfig)
from DeepCNNClassifer import logger

class ConfigurationManager:
    def __init__(
            self,
            config_path=CONFIG_FILE_PATH,
            params_path=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        logger.info(f"Setting up configuration for Data Ingestion Stage.")

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        logger.info(f"Configuration achieved - {data_ingestion_config}")
        return data_ingestion_config
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        
        logger.info(f"Setting up configuration for base model preparation stage.")
        
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model = Path(config.base_model),
            updated_base_model= Path(config.updated_base_model),
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_classes = self.params.CLASSES
        )
        logger.info(f"Configuration achieved - {prepare_base_model_config}")
        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbackConfig:

        logger.info(f"Setting up configuration for CallBack preparation stage.")

        config = self.config.prepare_callback

        model_checkpoint_dir = os.path.dirname(config.checkpoint_model_filepath)

        create_directories([
            Path(model_checkpoint_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbackConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath = Path(config.checkpoint_model_filepath)
        )
        logger.info(f"Configuration achieved - {prepare_callback_config}")
        return prepare_callback_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:

        logger.info(f"Setting up configuration for model training stage.")

        model_training_config = self.config.model_training
        prepare_base_model_config =  self.config.prepare_base_model
        params = self.params
        training_data_path = os.path.join(self.config.data_ingestion.unzip_dir, "PetImages")
        
        create_directories([model_training_config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = Path(model_training_config.root_dir),
            trained_model_path = Path(model_training_config.trained_model_path),
            updated_base_model_path = Path(prepare_base_model_config.updated_base_model),
            training_data = Path(training_data_path),
            params_epoch = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmented = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE
        )
        logger.info(f"Configuration achieved - {model_training_config}")
        return model_training_config

    def get_evaluation_config(self) -> ModelEvaluationConfig:

        logger.info(f"Setting up configuration for model evaluation stage.")

        model_evaluation_config = ModelEvaluationConfig(
            path = self.config.model_training.trained_model_path,
            training_data= self.config.data_ingestion.unzip_dir,
            params_img_size= self.params.IMAGE_SIZE,
            params_batch_size= self.params.BATCH_SIZE,
            mlflow_uri = MLFLOW_TRACKING_URI,
            all_params = self.params
        )
        logger.info(f"Configuration achieved - {model_evaluation_config}")
        return model_evaluation_config