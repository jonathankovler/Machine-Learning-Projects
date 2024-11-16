import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from preprocessing import filter_by_label

# Initialize Model and Tokenizer
CHECKPOINT = "avichr/heBERT_sentiment_analysis"

# Load Tokenizer and Model
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
MODEL = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=3)

data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)


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
        return TOKENIZER(examples["text"], truncation=True, padding="max_length", max_length=512)

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


# Function to filter by specific label
def filter_by_label(dataset, label):
    """
    Filters dataset based on label value.
    """
    return dataset.filter(lambda x: x['labels'] == label)


def balance_dataset(dataset, target_sizes):
    """
    Balance the dataset by selecting a target number of examples for each label.

    Args:
        dataset: The dataset to balance.
        target_sizes: Dictionary specifying the target number of examples for each label.

    Returns:
        balanced_dataset: A balanced dataset with specified target sizes for each label.
    """
    balanced_splits = []
    # Iterate over each label and filter the dataset, then select target number of examples
    for label_value, target_size in target_sizes.items():
        label_filtered_dataset = filter_by_label(dataset, label_value)
        # Shuffle and select the desired number of examples
        balanced_split = label_filtered_dataset.shuffle(seed=42).select(
            range(min(len(label_filtered_dataset), target_size)))
        balanced_splits.append(balanced_split)
    # Concatenate all balanced splits into one dataset
    balanced_dataset = concatenate_datasets(balanced_splits)
    return balanced_dataset


synthetic_processed_data = load_from_disk(
    "C:/Users/kovle/PycharmProjects/news-headlines-sentiment-analysis/data/processed_synthetic_data")

train_dataset, val_dataset, test_dataset = split_dataset(synthetic_processed_data)


def main():
    # Path to the preprocessed data file
    file_path = r'C:\\Users\\kovle\\PycharmProjects\\news-headlines-sentiment-analysis\\data\\synthetic_data_15k.csv'

    # Load the CSV file containing the preprocessed data
    dataset_preprocessed = pd.read_csv(file_path)

    # Columns to remove during preprocessing
    columns_to_remove = ["text", "Sentiment"]

    # Define the label mapping for this dataset
    label_mapping = {'Neutral': 0, 'Positive': 1, 'Negative': 2}

    # Convert filtered news data to Hugging Face dataset format
    dataset_preprocessed = Dataset.from_pandas(dataset_preprocessed)

    # Preprocess and tokenize the news dataset
    news_processed_data = preprocess_and_tokenize(dataset_preprocessed, label_mapping=label_mapping,
                                                  label_column="Sentiment",
                                                  columns_to_remove=columns_to_remove)

    # Save the preprocessed news dataset to disk
    news_processed_data.save_to_disk(
        "C:/Users/kovle/PycharmProjects/news-headlines-sentiment-analysis/data/processed_synthetic_data")

    # Load the unbalanced processed dataset from disk
    news_processed_data = load_from_disk(
        "C:/Users/kovle/PycharmProjects/news-headlines-sentiment-analysis/data/preprocessed_news_dataset")


if __name__ == "__main__":
    # main()
    # Load the unbalanced processed dataset from disk
    synthetic_processed_data = load_from_disk(
        "C:/Users/kovle/PycharmProjects/news-headlines-sentiment-analysis/data/processed_synthetic_data")

    train_dataset, val_dataset, test_dataset = split_dataset(synthetic_processed_data)
    '''

    # Split the balanced dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = split_dataset(balanced_news_dataset)

    print(train_dataset, val_dataset, test_dataset)
    '''
