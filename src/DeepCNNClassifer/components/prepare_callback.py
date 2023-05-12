import tensorflow as tf
from DeepCNNClassifer.entity.config_entity import PrepareCallbackConfig
from DeepCNNClassifer import logger
import time, os


class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config
    
    @property
    def _create_tb_callback(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f'tb_logs_at_{timestamp}'
        )
        logger.info(f'Saving TensorBoard Callbacks at {tb_running_log_dir}')
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
        
    
    @property
    def _create_ckpt_callback(self):
        logger.info(f'Saving Callbacks Checkpoints at {self.config.checkpoint_model_filepath}')
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )

    def get_tb_ckpt_callback(self):
        return [
            self._create_tb_callback,
            self._create_ckpt_callback
        ]