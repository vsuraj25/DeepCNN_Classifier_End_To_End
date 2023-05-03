from DeepCNNClassifer.components.prepare_base_model import PrepareBaseModel
from DeepCNNClassifer.config import ConfigurationManager
from DeepCNNClassifer import logger


def main():
    config = ConfigurationManager()
    prepare_base_model_config = config.get_base_model_config()
    prepare_base_model = PrepareBaseModel(config = prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Preparing Base Model... {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Base Model Preparation Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
