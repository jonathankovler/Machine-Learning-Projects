from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from configuration import PATH_TO_BASE_MODEL

# Load the fine-tuned model and tokenizer
model_path = "./heBERT-news-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)


base_model = AutoModelForSequenceClassification.from_pretrained(PATH_TO_BASE_MODEL)

# Create a sentiment analysis pipeline for the fine-tuned model
heBERT_news_sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=fine_tuned_model,
    tokenizer=tokenizer,
    return_all_scores=False,  # Return only the highest scoring label
    device=0  # Use GPU if available
)

# Create a sentiment analysis pipeline for the base model
heBERT = pipeline(
    "sentiment-analysis",
    model=PATH_TO_BASE_MODEL,
    tokenizer=tokenizer,
    return_all_scores=False,  # Return only the highest scoring label
    device=0  # Use GPU if available
)

# List of sentences to classify
texts = [
    "צהל חיסל מחבלים",  # "The IDF eliminated terrorists"
    "ישראל תקפה אווירית",  # "Israel launched an airstrike"
    "ישראל ניצחה",  # "Israel won"
    "חמאס תקף את ישראל",  # "Hamas attacked Israel"
    "אני שמח",  # "I'm happy"
    "איזה יום גרוע היה לי",  # "What a terrible day I had"
    "צהל הצליח לשמור על המדינה"  # "The IDF managed to protect the country"
]

# Iterate through each text and print predictions from both models
for text in texts:

    print(f"base model-->> text{text}, prediction{heBERT(text)}")
    print(f"fine-tuned model-->> text{text}, prediction{heBERT_news_sentiment_classifier(text)}")
    print()
