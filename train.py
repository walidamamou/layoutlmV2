
from ast import arg
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer,AutoTokenizer
from datasets import load_metric,load_from_disk
import os
import numpy as np
import torch
import warnings
import argparse

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--lr_scheduler_type",type=str,default="linear")
    parser.add_argument("--warmup_ratio",type=str,default=0.0)
    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_dir", type=str)

    args, _ = parser.parse_known_args()
    EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VALID_BATCH_SIZE = args.eval_batch_size
    LEARNING_RATE = float(args.learning_rate)
    LR_SCHEDULER_TYPE = args.lr_scheduler_type
    WARMUP_RATIO = float(args.warmup_ratio)

    os.makedirs(args.data_dir,exist_ok=True)
    os.makedirs(args.output_dir,exist_ok=True)

    # load datasets
    train_dataset = load_from_disk(f'{args.data_dir}train_split')
    valid_dataset = load_from_disk(f'{args.data_dir}eval_split')
    # Prepare model labels - useful in inference API

    labels = train_dataset.features["labels"].feature.names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    model = AutoModelForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                        num_labels=len(label2id))

    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')

    # Set id2label and label2id 
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Metrics
    metric = load_metric("seqeval")
    return_entity_level_metrics = True

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    args = TrainingArguments(
        output_dir=args.output_dir, # name of directory to store the checkpoints
        evaluation_strategy = "epoch",
        logging_strategy = "epoch",
        num_train_epochs=EPOCHS,
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        metric_for_best_model = "overall_f1",
        lr_scheduler_type = LR_SCHEDULER_TYPE,  #constant/linear/cosine/cosine_with_restarts/polynomial/constant_with_warmup
        warmup_ratio = WARMUP_RATIO, # optional, defaults to 0.0
        # max_steps=EPOCHS*len(train_dataloader),
        # fp16=True, # we use mixed precision (less memory consumption)
        save_total_limit = 1,
        load_best_model_at_end=True,
        save_strategy = "epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_result = trainer.evaluate(eval_dataset=valid_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    print(f"***** Eval results *****")
    for key, value in sorted(eval_result.items()):
        print(f"{key} = {value}\n")
    # trainer.save_model(args.output_dir)
    torch.save(model,args.output_dir+'.pth')
