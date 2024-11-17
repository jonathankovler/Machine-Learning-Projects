from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from configuration import PATH_TO_BASE_MODEL

# Load the fine-tuned model and tokenizer
model_path = "./heBERT-news-sentiment-classifier"  # The directory where you saved your model
tokenizer = AutoTokenizer.from_pretrained(model_path)
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

base_model = AutoModelForSequenceClassification.from_pretrained(PATH_TO_BASE_MODEL)

heBERT_news_sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=fine_tuned_model,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=0

)

heBERT = pipeline(
    "sentiment-analysis",
    model=PATH_TO_BASE_MODEL,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=0

)
# List of sentences to classify
texts = ["צהל חיסל מחבלים", "ישראל תקפה אווירית", "ישראל ניצחה", "חמאס תקף את ישראל", "אני שמח", "איזה יום גרוע היה לי",
         "צהל הצליח לשמור על המדינה"]

for text in texts:
    print(f"base model-->> text{text}, prediction{heBERT(text)}")
    print(f"fine-tuned model-->> text{text}, prediction{heBERT_news_sentiment_classifier(text)}")
    print()
