
def main():
    import os
    os.environ["WANDB_PROJECT"] = "pylate_mmarco_ar"
    import wandb 
    wandb.login(key="xxxxxxxxxxxxxx")

    import torch
    import pandas as pd
    from datasets import load_dataset, Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    from pylate import losses, models, utils

    import gc 
    from unicodedata import normalize
    #query_n = normalize('NFKC', query)
    def get_norm(example):
        example["text"] = normalize('NFKC', example["text"])
        return example
    gc.collect()
    documents = load_dataset(path="akhooli/ar_mmarco_5_docs")
    documents = documents.map(get_norm)

    gc.collect()
    queries = load_dataset(path="akhooli/ar_mmarco_5_queries")
    queries = queries.map(get_norm)

    gc.collect()
    train = load_dataset(path="akhooli/ar_mmarco_5_scores")

    # Set the transformation to load the documents/queries texts using the corresponding ids on the fly
    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents).transform,
    )

    # Define the base model, training parameters, and output directory
    model_name = "aubmindlab/bert-base-arabertv02"  # Choose the pre-trained model you want to use as base
    batch_size = 3
    num_train_epochs = 1
    # Set the run name for logging and output directory
    run_name = "kd_p5n"
    output_dir = f"/tmp/{run_name}"

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
        learning_rate=2e-5,
        save_steps=100, # default 500
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
    trainer.train()
    model.save_pretrained('ar_colbert5')
    # Your training code here
if __name__ == "__main__":
    main()


