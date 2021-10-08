from transformers import AutoModelForSequenceClassification, pipeline

from easy_transformers.loader import get_model_and_tokenizer, get_pipeline


def test_get_model_and_tokenizer_classification(model_name):
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        auto_model_type=AutoModelForSequenceClassification,
    )
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    prediction = classifier("this is good")
    assert prediction[0]["label"] == "LABEL_2"


def test_get_pipeline_classification(model_name):
    classifier = get_pipeline(
        "sentiment-analysis",
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        auto_model_type=AutoModelForSequenceClassification,
    )
    prediction = classifier("this is good")
    assert prediction[0]["label"] == "LABEL_2"
