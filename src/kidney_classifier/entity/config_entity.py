from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    raw_data_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_model_name: str

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    data_dir: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_val_split_size: float
    params_test_split_size: float
    # configs for callbacks
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    params_early_stopping_monitor: str
    params_early_stopping_patience: int
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    data_dir: Path
    all_params: dict
    params_image_size: list 
    params_batch_size: int
    params_val_split_size: float
    params_test_split_size: float