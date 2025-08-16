from dotenv import load_dotenv
from kidney_classifier.config.configuration import ConfigurationManager
from kidney_classifier.components.model_evaluation import Evaluation
from kidney_classifier import logger

STAGE_NAME = "MODEL EVALUATION"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        load_dotenv()
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluate_and_log()
    
if __name__ == "__main__":
    try:
        logger.info(f"-------- {STAGE_NAME} STARTED --------")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f"-------- {STAGE_NAME} FINISHED --------")
    
    except Exception as e:
        logger.exception(e)
        raise e