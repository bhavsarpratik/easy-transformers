from typing import List, Tuple, Union

import numpy as np
from cachetools import LRUCache, cached
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from easy_transformers import TEXT_EMB_CACHE_SIZE, TRANSFORMERS_CACHE_SIZE, constants
from easy_transformers.loggers import create_logger

logger = create_logger(project_name="easy_transformers", level="INFO")

# logger.info(f"EASY_TRANSFORMERS_CACHE_SIZE: {TRANSFORMERS_CACHE_SIZE}")
logger.info(f"EASY_TRANSFORMERS_TEXT_EMB_CACHE_SIZE: {TEXT_EMB_CACHE_SIZE}")


# @cached(LRUCache(maxsize=TRANSFORMERS_CACHE_SIZE))
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


# @cached(LRUCache(maxsize=TRANSFORMERS_CACHE_SIZE))
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


class EasySentenceTransformer:
    def __init__(self, model_name_or_path) -> SentenceTransformer:
        self.encoder = SentenceTransformer(model_name_or_path)

    @cached(LRUCache(maxsize=TEXT_EMB_CACHE_SIZE))
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
