from kidney_classifier import logger
from kidney_classifier.pipeline.data_ingestion import DataIngestionTrainingPipeline
from kidney_classifier.pipeline.prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "DATA INGESTION"

try:
    logger.info(f"-------- {STAGE_NAME} STARTED --------")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"-------- {STAGE_NAME} FINISHED --------")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "PREPARE BASE MODEL"

try:
    logger.info(f"-------- {STAGE_NAME} STARTED --------")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f"-------- {STAGE_NAME} FINISHED --------")
    
except Exception as e:
    logger.exception(e)
    raise e