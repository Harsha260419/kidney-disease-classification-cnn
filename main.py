from kidney_classifier import logger
from kidney_classifier.pipeline.data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "DATA INGESTION"

try:
    logger.info(f"-------- {STAGE_NAME} STARTED --------")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"-------- {STAGE_NAME} FINISHED --------")
    
except Exception as e:
    logger.exception(e)
    raise e