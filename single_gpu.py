#!/scratch/work/lunt1/.conda_envs/w2v2/bin/python
from pynvml import *
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    logging)
import torch 
import gc

# bits and bytes
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

# accelerator 
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

assert torch.cuda.is_available(), "Cuda is not available!"

#### Helper functions that print statistics 

def print_gpu_utilization(pretext=''):
    gc.collect()
    torch.cuda.empty_cache()

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"{pretext} GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

#### bits and bytes
def get_adam_bnb_optim(training_args):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim


##############
# experiments

def vanilla_training(model):
    print("------- Vanilla training")
    logging.set_verbosity_error()

    training_args = TrainingArguments(
        per_device_eval_batch_size=4, 
        **default_args, 
    )

    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=ds
    )
    result = trainer.train()
    print_summary(result)

def gradient_accumulation(model):
    print("\n------- Using gradient accumulation")
    training_args = TrainingArguments(
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=4, 
        **default_args
    )
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=ds
    )
    result = trainer.train()
    print_summary(result)

def gradient_checkpoint(model):
    print("\n------- Using gradient accumulation and checkpoint")
    training_args = TrainingArguments(
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True,
        **default_args
    )
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=ds
    )
    result = trainer.train()
    print_summary(result)

def mix_precision(model):
    print("\n------- Using mix precision training")
    training_args = TrainingArguments(
        per_device_eval_batch_size=4,
        fp16=True,
        **default_args
    )
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=ds
    )
    result = trainer.train()
    print_summary(result)

def mix_precision_combined(model):
    print("\n------- Using gradient accmulation, checkpoint and mix precision training")
    training_args = TrainingArguments(
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True,
        fp16=True,
        **default_args
    )
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=ds
    )
    result = trainer.train()
    print_summary(result)

def use_bitsandbytes(model):
    print("\n------- Using bits and bytes along with other settings")
    # training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        **default_args,
    )
    
    adam_bnb_optim = get_adam_bnb_optim(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds, 
        optimizers=(adam_bnb_optim, None)
    )

    result = trainer.train()
    print_summary(result)

def use_accelerate(model):
    print("\n------- Using bits and bytes AND accelerate, along with other settings")

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        **default_args,
    )

    dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    adam_bnb_optim = get_adam_bnb_optim(training_args)
    accelerator = Accelerator(mixed_precision='fp16')
    model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

    model.train()
    for step, batch in enumerate(dataloader, start=1):
        loss = model(**batch).loss
        loss = loss/training_args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    print_gpu_utilization()

if __name__ == "__main__":

    #### create dummy dataset

    seq_len, dataset_size = 512, 512
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)), 
        "labels": np.random.randint(0, 1, (dataset_size)),
    }

    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")

    print_gpu_utilization("Before loading anything.") # 110MB

    #### load a tiny tensor to cuda 
    torch.ones((1,1)).to("cuda")
    print_gpu_utilization("After loading a tiny tenor.") # 728 MB

    #### load a modal 
    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
    print_gpu_utilization("After loading the model.")

    #### default args used across all experiments 
    default_args = {
        "output_dir": "tmp",
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "log_level": "error",
        "report_to": "none",
    }

    # vanilla_training(model)
    # gradient_accumulation(model)
    # gradient_checkpoint(model)
    # mix_precision(model)
    # mix_precision_combined(model)
    # use_bitsandbytes(model)
    use_accelerate(model)