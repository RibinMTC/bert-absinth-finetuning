<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Absinth Bert Fine-Tuning Example
This repository has been modified from the [HuggingFace repository for text classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

The following example fine-tunes multilingual BERT on [Absinth Hallucination Detection Task](https://github.com/mediatechnologycenter/Absinth).
- Install requirements:
  ```bash
    pip3 install requirements.txt
    ```
  - By default, the Absinth dataset does not contain the source articles, therefore perform the following steps to add the source articles as an additional column to the dataset:
    - Download the source articles from [here](https://drive.google.com/file/d/1taGM6qToFDB37RjU5BjlYtiup_CYpvXZ/view), unzip and place it in the [data](./data) folder. 
    - Run the following script to merge the source articles and save the dataset locally:
      ```bash
      python add_source_articles_and_save_locally.py
      ```
      This will store the dataset locally in [data](./data).

- We can now finetune BERT on the Absinth dataset: 
  - The `text_column_name` contain the column name of the article followed by the column name of the summary sentence. 
  - Please note that the article is truncated to fit into the specified `max_seq_length`.
  - Please set `output_dir` to the location where the trained model and training logs should be saved.
  ```bash
  python run_classification.py \
      --model_name_or_path  google-bert/bert-base-multilingual-uncased \
      --dataset_name ./data/absinth_dataset_with_source_articles \
      --shuffle_train_dataset \
      --text_column_name lead_with_article,text \
      --label_column_name label \
      --logging_steps 50 \
      --evaluation_strategy steps \
      --eval_steps 100 \
      --do_train \
      --do_eval \
      --max_seq_length 512 \
      --per_device_train_batch_size 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --save_steps -1 \
      --output_dir /output_dir
  ```

