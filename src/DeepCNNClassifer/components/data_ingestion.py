import os,sys
from pathlib import Path
from DeepCNNClassifer.entity.config_entity import DataIngestionConfig
from DeepCNNClassifer.utils import *
import urllib.request as request
from zipfile import ZipFile
from DeepCNNClassifer import logger
from tqdm import tqdm


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        logger.info(f"{'>>>>'*5} Getting Configurations {'<<<<'*5}.")
        self.config = config

    def download_file(self):
        logger.info(f"{'>>>>'*5} Downloading the zip file from the url - {self.config.source_url} to file - {self.config.local_data_file} {'<<<<'*5}.")
        if not os.path.exists(self.config.local_data_file):
            filename, headers= request.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"{'>>>>'*5} Zip File Downloaded and saved at {self.config.local_data_file} {'<<<<'*5}.")
        else:
            logger.info(f"{'>>>>'*5} File already present in {self.config.local_data_file} {'<<<<'*5} of size {get_size(Path(self.config.local_data_file))}.")
        

    def _preprocess(self, zf : ZipFile, f : str, working_dir : str):
        target_filepath = os.path.join(working_dir, f)
        
        if not os.path.exists(target_filepath):
            zf.extract(f, working_dir)
        
        if os.path.getsize(target_filepath) == 0:
            os.remove(target_filepath)
            logger.info(f"{'>>>>'*5} Removing file {target_filepath} {'<<<<'*5}.")

    def _get_updated_list_of_files(self, list_of_files):
        return [f for f in list_of_files if f.endswith('.jpg') and ('Cat' in f or 'Dog' in f)]


    def unzip_and_clean(self):
        logger.info(f"{'>>>>'*5} Extracting the zip file.{'<<<<'*5}.")
        with ZipFile(file = self.config.local_data_file, mode="r") as zf:
            list_of_files = zf.namelist()
            update_list_of_files = self._get_updated_list_of_files(list_of_files)
            for f in tqdm(update_list_of_files):
                self._preprocess(zf, f, self.config.unzip_dir)
        logger.info(f"{'>>>>'*5} Successfully extracted data from the zip file and saved a location - {self.config.unzip_dir} {'<<<<'*5}.")