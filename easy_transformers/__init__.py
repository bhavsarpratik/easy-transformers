import os

from easy_transformers import constants

TRANSFORMERS_CACHE_SIZE = int(
    os.environ.get(
        "EASY_TRANSFORMERS_CACHE_SIZE", constants.DEFAULT_EASY_TRANSFORMERS_CACHE_SIZE
    )
)

TEXT_EMB_CACHE_SIZE = int(
    os.environ.get(
        "EASY_TRANSFORMERS_TEXT_EMB_CACHE_SIZE",
        constants.DEFAULT_EASY_TRANSFORMERS_TEXT_EMB_CACHE_SIZE,
    )
)
