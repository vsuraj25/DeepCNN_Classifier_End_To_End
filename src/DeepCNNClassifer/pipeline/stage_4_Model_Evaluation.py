from DeepCNNClassifer.components.evaluate_model import ModelEvaluation
from DeepCNNClassifer.config import ConfigurationManager
from DeepCNNClassifer import logger
import tensorflow as tf

def main():
    config = ConfigurationManager()
    validation_config = config.get_evaluation_config()
    evaluation = ModelEvaluation(validation_config)
    evaluation.evaluate()
    evaluation.save_scores()

if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Model Evaluation started... {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Model Evaluation Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
