import os

from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd


def merge_source_articles_with_dataset() -> DatasetDict:
    source_articles_file_path = "data/absinth_source_articles.jsonl"
    article_id_key_name = "article_id"
    splits = ["train", "validation", "test"]

    if not os.path.exists(source_articles_file_path):
        raise FileNotFoundError("No absinth_source_articles.jsonl file found! Please download source articles first.")

    source_articles_df = pd.read_json(source_articles_file_path, lines=True)

    absinth_dataset = load_dataset("mtc/absinth_german_faithfulness_detection_dataset")
    for split in splits:
        absinth_dataset_df = absinth_dataset[split].to_pandas()
        absinth_with_source_articles_df = pd.merge(absinth_dataset_df, source_articles_df, how="left",
                                                   on=article_id_key_name).reset_index(drop=True)
        absinth_dataset[split] = Dataset.from_pandas(absinth_with_source_articles_df)

    return absinth_dataset


def store_dataset_locally(absinth_dataset_with_source_articles: DatasetDict):
    save_path = './data/absinth_dataset_with_source_articles'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    absinth_dataset_with_source_articles.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")


def main():
    absinth_with_source_articles = merge_source_articles_with_dataset()
    store_dataset_locally(absinth_with_source_articles)


if __name__ == '__main__':
    main()
