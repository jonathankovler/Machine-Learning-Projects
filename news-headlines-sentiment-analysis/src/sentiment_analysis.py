from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import configuration
import torch
import transformers

# Determine device (GPU or CPU)
device_id = 0 if torch.cuda.is_available() else -1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = transformers.BertTokenizerFast.from_pretrained(configuration.PATH_TO_TOKENIZER)
model = transformers.BertForSequenceClassification.from_pretrained(configuration.PATH_TO_MY_MODEL).to(device)

# Create sentiment analysis pipeline
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=device_id
)

# Load data
df = pd.read_csv(
    configuration.PATH_TO_TABLE,
    encoding='utf-8-sig'
)


# Function to add sentiment to the csv file
def analyze_sentiment_and_append(row):
    headline = row['Headline']
    sent_analysis = sentiment_analysis(headline)[0]
    row['Sentiment'] = sent_analysis['label']
    row['Score'] = sent_analysis['score']
    return row


if __name__ == '__main__':
    try:
        # Apply sentiment analysis row by row
        df['Sentiment'] = None
        df['Score'] = None

        for idx, row in df.iterrows():
            headline = row['Headline']
            sent_analysis = sentiment_analysis(headline)[0]
            df.at[idx, 'Sentiment'] = sent_analysis['label']
            df.at[idx, 'Score'] = sent_analysis['score']

        # Save processed DataFrame to CSV
        df.to_csv(configuration.PATH_TO_PROCESSED_TABLE, index=False, encoding='utf-8-sig')
        print("created")

    except Exception as ex:
        print(f"An error occurred: {ex}")
