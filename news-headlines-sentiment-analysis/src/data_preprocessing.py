import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from preprocessing import filter_by_label

# Initialize Model and Tokenizer
CHECKPOINT = "avichr/heBERT_sentiment_analysis"

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Function to filter by specific label
def filter_by_label(dataset, label):
    """
    Filters dataset based on label value.
    """
    return dataset.filter(lambda x: x['labels'] == label)


def preprocess_and_tokenize(dataset, columns_to_remove, label_mapping, label_column='label_column'):
    """
    Preprocess the dataset by tokenizing the text, mapping labels, and removing unwanted columns.

    Args:
        dataset: The dataset to preprocess.
        label_mapping: Dictionary mapping labels to numerical values.
        label_column: The name of the label column in the dataset.

    Returns:
        processed_dataset: The processed dataset ready for model input.
    """

    # Tokenize Text
    def tokenize_text(examples):
        # Tokenize the text column, truncate to max length of 512, and add padding
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Transform labels
    def transform_labels(example):
        # Map the label in label_column to its corresponding numerical value
        example["labels"] = label_mapping[example[label_column]]
        return example

    # Apply tokenization and label transformation to the dataset
    processed_dataset = dataset.map(tokenize_text, batched=True)
    processed_dataset = processed_dataset.map(transform_labels)
    # Remove unnecessary columns from the dataset
    processed_dataset = processed_dataset.remove_columns(columns_to_remove)

    return processed_dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        dataset: The dataset to split.
        train_ratio: The proportion of the data to use for training.
        val_ratio: The proportion of the data to use for validation.
        seed: Seed for shuffling.

    Returns:
        train_dataset, val_dataset, test_dataset: The split datasets.
    """
    # Shuffle the dataset to ensure random distribution of samples
    shuffled_dataset = dataset.shuffle(seed=seed)
    total_size = len(shuffled_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split the dataset into training, validation, and test sets
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
    test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))

    return train_dataset, val_dataset, test_dataset


def prepare_new_data(new_data, label_mapping, label_column_name):
    """
    Prepare new data to be used with the existing dataset, including tokenization and label transformation.

    Args:
        new_data: A pandas DataFrame containing new data.
        label_mapping: Dictionary mapping labels to numerical values.
        label_column_name: The name of the label column in the new data.

    Returns:
        new_dataset_tokenized: A tokenized Hugging Face dataset ready for use.
    """
    # Convert the pandas DataFrame to a Hugging Face dataset
    dataset = datasets.Dataset.from_pandas(new_data)
    # Preprocess and tokenize the dataset
    return preprocess_and_tokenize(dataset, label_mapping, label_column=label_column_name)


if __name__ == "__main__":
    # Path to the preprocessed data file
    preprocessed_data = r'C:\Users\kovle\PycharmProjects\news-headlines-sentiment-analysis\data\sentiments-dataset.csv'

    # Load the CSV file containing the preprocessed data
    dataset = pd.read_csv(preprocessed_data)

    # Filter out rows that belong to the 'NEWS' category
    news_dataset = dataset[dataset["category"] == "NEWS"]

    # Columns to remove during preprocessing
    columns_to_remove = ['id', 'category', 'class', 'total_tags', 'selected_tag',
                         'polarity', 'text', '__index_level_0__', 'tag']

    # Define the label mapping for this dataset
    label_mapping = {'נ': 0, 'ח': 1, 'ש': 2}  # Example of a different label mapping

    # Convert filtered news data to Hugging Face dataset format
    news_dataset = Dataset.from_pandas(news_dataset)

    # Preprocess and tokenize the news dataset
    news_processed_data = preprocess_and_tokenize(news_dataset, label_mapping=label_mapping, label_column='tag',
                                                  columns_to_remove=columns_to_remove)

    # Save the preprocessed news dataset to disk
    news_processed_data.save_to_disk(
        "C:/Users/kovle/PycharmProjects/news-headlines-sentiment-analysis/data/preprocessed_news_dataset")
