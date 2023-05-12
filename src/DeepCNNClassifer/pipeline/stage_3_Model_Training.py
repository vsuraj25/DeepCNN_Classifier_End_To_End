from DeepCNNClassifer.components.prepare_callback import PrepareCallback
from DeepCNNClassifer.components.model_training import ModelTraining
from DeepCNNClassifer.config import ConfigurationManager
from DeepCNNClassifer import logger

def main():
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_callback_config()
    prepare_base_model = PrepareCallback(config = prepare_base_model_config)
    callback_list = prepare_base_model.get_tb_ckpt_callback()

    model_training_config = config.get_model_training_config()
    model_training = ModelTraining(config = model_training_config)
    model_training.get_base_model()
    model_training.train_valid_generator()
    model_training.train(
        callback_list = callback_list
    )

if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Model Training started... {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Model Training Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
