from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model: Path
    updated_base_model: Path
    params_image_size : list
    params_learning_rate : float
    params_include_top : bool
    params_weights : str
    params_classes : int

@dataclass(frozen=True)
class PrepareCallbackConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    prediction_model_path : Path
    updated_base_model_path : Path
    training_data : Path
    params_epoch : int
    params_batch_size : int
    params_is_augmented : bool
    params_image_size : list

@dataclass(frozen=True)
class ModelEvaluationConfig:
    path : Path
    training_data : Path
    params_img_size : list
    params_batch_size : int
    mlflow_uri : str
    all_params : dict
