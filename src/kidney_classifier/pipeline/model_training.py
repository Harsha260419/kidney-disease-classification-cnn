from kidney_classifier.config.configuration import ConfigurationManager
from kidney_classifier.components.model_training import Training
from kidney_classifier import logger

STAGE_NAME = "TRAINING"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_split()
        training.train()   
    

if __name__ == "__main__":
    try:
        logger.info(f"-------- {STAGE_NAME} STARTED --------")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"-------- {STAGE_NAME} FINISHED --------")
    
    except Exception as e:
        logger.exception(e)
        raise e