import os
from kidney_classifier.constants import *
from kidney_classifier.utils.common import create_directories, read_yaml
from kidney_classifier.entity.config_entity import (DataIngestionConfig, 
                                                      PrepareBaseModelConfig, 
                                                      TrainingConfig,
                                                      EvaluationConfig)
from pathlib import Path

class ConfigurationManager:

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            raw_data_dir=Path(config.raw_data_dir)
        )
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        model_name = self.params.MODEL_NAME
        create_directories([config.root_dir])
        
        base_model_path = Path(config.root_dir) / f"base_model_{model_name}.h5"
        updated_base_model_path = Path(config.root_dir) / f"updated_base_model_{model_name}.h5"
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=base_model_path,
            updated_base_model_path=updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_model_name=model_name
        )
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        model_name = self.params.MODEL_NAME
        prepare_base_model_config = self.get_prepare_base_model_config()
        updated_base_model_path = prepare_base_model_config.updated_base_model_path
        
        params = self.params
        data_dir = self.config.data_ingestion.raw_data_dir
        tensorboard_root_logdir = Path(training.root_dir) / "tensorboard_logs"
        create_directories([Path(training.root_dir), tensorboard_root_logdir])

        trained_model_path = Path(training.root_dir) / f"model_{model_name}.h5"
        checkpoint_model_path = Path(training.root_dir) / f"best_model_{model_name}.h5"
        
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=trained_model_path,
            updated_base_model_path=updated_base_model_path,
            data_dir=Path(data_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_val_split_size=params.VAL_SPLIT_SIZE,
            params_test_split_size=params.TEST_SPLIT_SIZE,
            # callback configs
            tensorboard_root_log_dir=tensorboard_root_logdir,
            checkpoint_model_filepath=checkpoint_model_path,
            params_early_stopping_monitor=params.EARLY_STOPPING_MONITOR,
            params_early_stopping_patience=params.EARLY_STOPPING_PATIENCE
        )

        return training_config
    
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation
        model_name = self.params.MODEL_NAME
        params = self.params
        data_dir = self.config.data_ingestion.raw_data_dir
        path_of_model = self.get_training_config().checkpoint_model_filepath

        evaluation_config = EvaluationConfig(
            root_dir=Path(eval_config.root_dir),
            path_of_model=Path(path_of_model),
            data_dir=Path(data_dir),
            all_params=params,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_val_split_size=params.VAL_SPLIT_SIZE,
            params_test_split_size=params.TEST_SPLIT_SIZE
        )
        create_directories([evaluation_config.root_dir])
        return evaluation_config