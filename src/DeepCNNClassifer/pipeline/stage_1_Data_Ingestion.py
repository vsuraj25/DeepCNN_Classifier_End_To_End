from DeepCNNClassifer.components import DataIngestion
from DeepCNNClassifer.config import ConfigurationManager
from DeepCNNClassifer import logger
def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config = data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.unzip_and_clean()


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Data Ingestion. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Data Ingestion Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
