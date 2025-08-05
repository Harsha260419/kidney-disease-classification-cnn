import os
import zipfile
import gdown
from kidney_classifier import logger
from kidney_classifier.utils.common import get_size
from kidney_classifier.entity.config_entity import DataIngestionConfig

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        try:
            dataset_url = self.config.source_URL
            zip_download_dir_path = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir_path), exist_ok=True)

            if not os.path.exists(zip_download_dir_path):
                logger.info(f"Downloading data from {dataset_url} into {zip_download_dir_path}")
                file_id = dataset_url.split("/")[-2]
                prefix = "https://drive.google.com/uc?export=download&id="
                gdown.download(prefix + file_id, zip_download_dir_path)
                logger.info(f"Downloaded data to {zip_download_dir_path}")
            else:
                logger.info(f"File already exists at {zip_download_dir_path}, skipping download.")
        
        except Exception as e:
            raise e

    def unzip_data(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        
        if len(os.listdir(unzip_path)) == 0:
            logger.info(f"Unzipping file {self.config.local_data_file} into {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as f:
                f.extractall(unzip_path)
            logger.info(f"Unzipped data to {unzip_path}")
        else:
            logger.info(f"Data already exists in {unzip_path}, skipping unzip.")
