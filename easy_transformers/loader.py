from typing import List, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from easy_transformers import constants
from easy_transformers.aws_utils import download_model_from_s3
from easy_transformers.loggers import create_logger

logger = create_logger(project_name="easy_transformers", level="INFO")


def get_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    auto_model_type: _BaseAutoModelClass,
    max_length: int = constants.DEFAULT_MAX_LENGTH,
    auto_model_config: AutoConfig = None,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Get transformer model and tokenizer

    Args:
        model_name_or_path (str): model name
        tokenizer_name_or_path (str): tokenizer name
        auto_model_type (_BaseAutoModelClass): auto model object such as AutoModelForSequenceClassification
        max_length (int): max length of text
        auto_model_config (AutoConfig): AutoConfig object

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: model and tokenizer
    """
    logger.info(f"Loading model: {model_name_or_path}")
    if auto_model_config:
        model = auto_model_type.from_pretrained(
            model_name_or_path, config=auto_model_config
        )
    else:
        model = auto_model_type.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, max_length=max_length
    )
    return model, tokenizer


def get_pipeline(
    pipeline_name: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    auto_model_type: _BaseAutoModelClass,
    max_length: int = 512,
    auto_model_config: AutoConfig = None,
    return_all_scores: bool = False,
) -> pipeline:
    """Get transformer pipeline

    Args:
        pipeline_name (str): transformer pipeline name
        model_name_or_path (str): model name
        tokenizer_name_or_path (str): tokenizer name
        auto_model_type (_BaseAutoModelClass): auto model object such as AutoModelForSequenceClassification
        max_length (int): max length of text
        auto_model_config (AutoConfig): AutoConfig object
        return_all_scores (bool): whether to return all scores

    Returns:
        pipeline: transformer pipeline
    """
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        auto_model_config=auto_model_config,
        max_length=max_length,
        auto_model_type=auto_model_type,
    )
    return pipeline(
        pipeline_name,
        model=model,
        tokenizer=tokenizer,
        return_all_scores=return_all_scores,
    )


class ONNXPipelineForSequenceClassification:

    """Custom ONNX Runtime Pipeline for Sequence Classification (same as the HuggingFace Pipeline)"""

    """
    Args : 
        model_path : path of the onnx model
        tokenizer_path : path to the tokenizer (model directory)
        label_map (dict): the label mappings between labels and strings
    """

    def __init__(self, model_path: str, tokenizer_path: str, label_map: dict):
        """Load the ONNX model runtime and the tokenizer with label map being the labels to class names mapping"""

        self.ort_session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        self.label_map = label_map

    def predict(self, text: List[str]) -> List[dict]:
        """
        Args:
            text : Batch of sentences to extract classification labels from
        Returns:
            Tokenizes and predicts the input texts using the ONNX runtime.
            and the scores and labels for each sample
        """
        messages = self.tokenizer(text, padding=True, return_tensors="np")
        logits = self.ort_session.run(["logits"], dict(messages))[0]
        scores = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1).numpy()
        inds = np.argmax(scores, axis=1)

        predictions = []
        for i in range(len(logits)):
            predictions.append(
                {"label": self.label_map[int(inds[i])], "score": scores[i][inds[i]]}
            )
        return predictions


class ONNXSequenceClassificationModel:
    """
    Downloads the model from given remote_dir(s3 url) and loads the model pipeline using the ONNXPipelineForSequenceClassification
    Args:
        model_name: The name of folder you want to download the model
        remote_dir: S3 url where the model directory is present
        onn_name: the name of the onnx file in the model directory
        label_map: dictionary mapping from labels to strings
    """

    def __init__(
        self, model_name: str, remote_dir: str, onnx_name: str, label_map: dict
    ):
        logger.info("Loading Sentence Sentiment Model...")
        self.model = self.load_onnx_sequence_classification_pipeline(
            model_name, remote_dir, onnx_name, label_map
        )

    def load_onnx_sequence_classification_pipeline(
        self, model_name: str, remote_dir: str, onnx_name: str, label_map: dict
    ) -> ONNXPipelineForSequenceClassification:
        """Downloads the sentence sentiment model from s3 and loads it."""

        model_path = download_model_from_s3(
            model_name=model_name, remote_dir=remote_dir
        )

        logger.info("Loading Sentiment Model...")
        model_path = model_path + "/" + onnx_name
        tokenizer_path = model_path

        pipeline = ONNXPipelineForSequenceClassification(
            model_path, tokenizer_path, label_map
        )

        return pipeline


class EasySentenceTransformer:
    def __init__(self, model_name_or_path) -> SentenceTransformer:
        self.encoder = SentenceTransformer(model_name_or_path)

    def encode(
        self, text: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs
    ) -> np.array:
        """Get unit normalised text embedding

        Args:
            text (str): sentence
            normalize_embeddings (str): whether to normalise embeddings to unit vector

        Returns:
            np.array: sentence emb
        """
        return self.encoder.encode(
            text, normalize_embeddings=normalize_embeddings, **kwargs
        )
