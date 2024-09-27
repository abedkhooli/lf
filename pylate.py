import os
os.environ["WANDB_PROJECT"] = "pylate_mmarco_ar"
import wandb 
wandb.login(key="5c5bd3b5c27fad8669de36ad478b24d1aa8625e4")

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import losses, models, utils

import gc 
gc.collect()
#documents = Dataset.from_pandas(df_docs, preserve_index=False)
documents = load_dataset("csv", data_files='common-docs.tsv', sep='\t')

gc.collect()
queries = load_dataset("csv", data_files='common-queries.tsv', sep='\t')

train = load_dataset(path="akhooli/arabic_mmarco_scores")

# Set the transformation to load the documents/queries texts using the corresponding ids on the fly
train.set_transform(
    utils.KDProcessing(queries=queries, documents=documents).transform,
)

# Define the base model, training parameters, and output directory
model_name = "aubmindlab/bert-base-arabertv02"  # Choose the pre-trained model you want to use as base
batch_size = 2
num_train_epochs = 1
# Set the run name for logging and output directory
run_name = "kd-bert"
output_dir = f"output/{run_name}"

# Initialize the ColBERT model from the base model
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model to make the training faster
model = torch.compile(model)

# Configure the training arguments (e.g., epochs, batch size, learning rate)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,
    learning_rate=1e-5,
)

# Use the Distillation loss function for training
train_loss = losses.Distillation(model=model)

# Initialize the trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=train_loss,
    data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
)
def main():
    trainer.train()
    # Your training code here
if __name__ == "__main__":
    main()


