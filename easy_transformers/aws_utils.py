import os
import shutil

from cloudpathlib import CloudPath

from easy_transformers import loggers

logger = loggers.create_logger(
    project_name="aws_utils", level="INFO", json_logging=True
)


def download_s3_folder(uri: str, local_dir: str) -> None:
    """
    Download the contents of a folder directory
    Args:
        uri: Remote S3 model folder url
        local_dir: directory path in the local file system
    """
    cp = CloudPath(uri)
    logger.info(f"Downloading...")
    cp.download_to(local_dir)
    logger.info("Download complete!")


def download_model_from_s3(
    model_name: str, remote_dir: str, download_folder: str = "./models"
) -> str:
    """Downloads model from s3 bucket based on given arguments and saves in default download folder

    Args:
        model_name (str): Name of model to be loaded
        pipeline_name (str): Pipeline name like sentiment, absa, etc
        download_folder (str)(Optional): Path where model will be downloaded
        version (str)(Optional): Version of model if any

    Return:
        str: Path where model is downloaded
    """

    model_path = f"{download_folder}/{model_name}"
    os.makedirs(model_path, exist_ok=True)

    ## If the folder already contains model files return model_path
    if os.listdir(model_path):
        logger.info(f"Model already exists at {model_path} skipping the downloading...")
        return model_path

    shutil.rmtree(model_path)

    logger.info(f"Downloading the model from: {remote_dir} to {model_path}")
    download_s3_folder(
        uri=remote_dir,
        local_dir=model_path,
    )

    return model_path
