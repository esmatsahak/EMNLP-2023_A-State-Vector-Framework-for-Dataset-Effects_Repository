import numpy as np 
import torch
import torch.nn as nn
import transformers
import datasets
import logging
#logging.basicConfig(level=logging.INFO)
import os
import pickle
import json
import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict, Optional
from transformers import default_data_collator
import wandb
import argparse
from tqdm import tqdm
from functools import partialmethod
import shutil
from pathlib import Path
from utils import init_or_resume_wandb_run

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        # Setting MultitaskModel up as a PretrainedModel allows us to take advantage of Trainer features
        super().__init__(transformers.PretrainedConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        # Creates a MultitaskModel using the model class and config objects from single-task models
        # Do this by creating each single-task model and having them share the same encoder transformer
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name]
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)
  
    @classmethod
    def get_encoder_attr_name(cls, model):
        # Encoder transformer is named differently in each model architecture
        # This method gets the name of the encoder attribute
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")
  
    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

class NLPDataCollator():
    # Extending existing DataCollator to work with NLP dataset batches
    def __call__(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                else:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
          # otherwise, revert to using the default collate_batch
          return default_data_collator(features)

class StrIgnoreDevice(str):
    # A hack to prevent error: Trainer is going to call .to(device) on every input, but we need to pass in task_name string
    def to(self, device):
        return self

class DataLoaderWithTaskname:
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset
  
    def __len__(self):
        return len(self.data_loader)
  
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch

class MultitaskDataloader:
    # Data loader that combines and samples from multiple single-task data loaders
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        # For each batch, sample a task, and yield a batch from the respective task dataloader
        # Using size-proportional sampling
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

class MultitaskTrainer(transformers.Trainer):
    def get_single_dataloader(self, task_name, dataset, mode):
        if mode == "train":
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset")
            batch_size = self.args.train_batch_size
        elif mode == "eval":
            if self.eval_dataset is None:
                raise ValueError("Trainer: training requires a eval_dataset")
            batch_size = self.args.train_batch_size
        else:
            raise ValueError("Invalid mode")
        sampler = (
            RandomSampler(dataset)
            if self.args.local_rank == -1
            else DistributedSampler(dataset)
        )
        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self.data_collator
            )
        )
        return data_loader

    def get_train_dataloader(self):
        # Returns a MultitaskDataloader, which is not actually a Dataloader but an iterable that returns a generator that samples from each task Dataloader
        return MultitaskDataloader({
            task_name: self.get_single_dataloader(task_name, task_dataset, "train")
            for task_name, task_dataset in self.train_dataset.items()
        })
    
    def get_eval_dataloader(self, eval_dataset):
        # Returns a MultitaskDataloader, which is not actually a Dataloader but an iterable that returns a generator that samples from each task Dataloader
        return MultitaskDataloader({
            task_name: self.get_single_dataloader(task_name, task_dataset, "eval")
            for task_name, task_dataset in self.eval_dataset.items()
        })

def convert_to_cola_features(example_batch):
    inputs = example_batch['sentence']
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_sst2_features(example_batch):
    inputs = example_batch['sentence']
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_mrpc_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_stsb_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_qnli_features(example_batch):
    inputs = list(zip(example_batch['question'], example_batch['sentence']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rte_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("exp_id", type=int, help="experiment ID")
parser.add_argument("seed", type=int, choices=[42,1,1234,123,10])
parser.add_argument("enc", type=str, choices=['bert', "roberta"], help='encoder')
parser.add_argument("model", type=str, choices=['bert-base-cased', "roberta-base"], help="model name")
args = parser.parse_args()
print(args)
exp_id = args.exp_id
encoder = args.enc
model_name = args.model
seed = args.seed
job_id = int(os.environ["SLURM_JOB_ID"]) # Set to 1 when running locally

np.random.seed(seed)
torch.manual_seed(seed)

experiments = [
    "100000", "010000", "001000", "000100", "000010", "000001", 
    "110000", "101000", "100100", "100010", "100001", "011000", 
    "010100", "010010", "010001", "001100", "001010", "001001",
    "000110", "000101", "000011", "101010", "101001", "100110",
    "100101", "011010", "011001", "010110", "010101", "111100",
    "110011", "001111", "111111"]
all_tasks = [["cola", 2], ["sst2", 2], ["mrpc", 2], ["stsb", 1], ["qnli", 2], ["rte", 2]]
max_length = 128
train_size = 2490 # 50 
#val_size = 50
lr = 2e-5
warmup_ratio = 0.1
lr_scheduler = "constant_with_warmup"
batch_size = 32
epochs = 3
weight_decay = 0.01
dropout = 0.1
attention_dropout = 0.1
classifier_dropout = 0.1
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.99
max_grad_norm = 1.0
steps = 6 

exp = experiments[exp_id-1]
exp_dir = f"../experiments/{encoder}/{seed}/{exp_id}"
output_dir = f"{exp_dir}/checkpoints"

if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
 
# wandb.login(key="4579cc2ec3bcbfd7de749970537a0ced71f36121")
# wandb.init(dir=exp_dir, project="multitask_nlp", entity="ziningzhu", group=f"{encoder}_{seed}", name=f"exp {exp_id}", resume=True, id=f"{encoder}_{seed}_{exp_id}_{job_id}")
cfg = init_or_resume_wandb_run(exp_dir, Path(exp_dir, "wandb_run_id.txt"), project_name="multitask_nlp", entity="ziningzhu", run_name=f"exp {exp_id}")

tasks = []
for i in range(len(exp)):
    if exp[i] == '1':
        tasks += [all_tasks[i]]
print('Tasks: ', [task[0] for task in tasks])

dataset_dict = {
    task[0]: datasets.load_dataset('glue', name=task[0])
    for task in tasks
}

for task_name in dataset_dict:
    dataset_dict[task_name]["train"] = dataset_dict[task_name]["train"].shuffle(seed=seed).select(range(train_size))
    #dataset_dict[task_name]["validation"] = dataset_dict[task_name]["validation"].shuffle(seed=seed).select(range(val_size))

for task_name, dataset in dataset_dict.items():
    print(task_name)
    print(dataset_dict[task_name]["train"][0])
    print()

model_type_dict = {
    task_name: transformers.AutoModelForSequenceClassification
    for task_name in dataset_dict
}
model_config_dict = {
    task[0]: transformers.AutoConfig.from_pretrained(model_name, hidden_dropout_prob=dropout, attention_probs_dropout_prob=attention_dropout, classifier_dropout=classifier_dropout, num_labels=task[1])
    for task in tasks
}

multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict=model_type_dict,
    model_config_dict=model_config_dict
)

print('model encoder')
print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
print()
for task_name, model in multitask_model.taskmodels_dict.items():
    print(task_name)
    if encoder == "bert":
        print(model.bert.embeddings.word_embeddings.weight.data_ptr())
    else:
        print(model.roberta.embeddings.word_embeddings.weight.data_ptr())
    print()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

convert_func_dict = {
    task_name: eval("convert_to_" + task_name + "_features")
    for task_name in dataset_dict
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False
        )
        features_dict[task_name][phase].set_format(
            type="torch",
            columns=['input_ids', 'attention_mask', 'labels']
        )

train_dataset = {
    task_name: dataset["train"]
    for task_name, dataset in features_dict.items()
}
eval_dataset = {
    task_name: dataset["validation"]
    for task_name, dataset in features_dict.items()
}

trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=beta1,
        adam_beta2=beta2,
        adam_epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        num_train_epochs=epochs,
        lr_scheduler_type=lr_scheduler, 
        warmup_ratio=warmup_ratio,
        save_strategy="steps",
        logging_steps=steps,
        save_steps=steps,
        save_total_limit=2,
        seed=RANDOM_SEED,
        eval_steps=steps,
        remove_unused_columns=False,
        label_names=['labels'],
        load_best_model_at_end=True,
        report_to="wandb",
        disable_tqdm=True
    ),
    data_collator=NLPDataCollator(),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

if len(os.listdir(trainer.args.to_dict()["output_dir"]))==0:
    print("Starting from scratch")
    trainer.train()
else:
    print("Resuming from checkpoint")
    trainer.train(resume_from_checkpoint=True)

wandb.finish()
shutil.rmtree(f"{exp_dir}/wandb", ignore_errors=True)
shutil.rmtree(output_dir, ignore_errors=True)

multitask_model.encoder.save_pretrained(f"{exp_dir}/encoder")

preds_dict = {}
for task_name in dataset_dict:
    eval_dataloader = DataLoaderWithTaskname(
            task_name,
            super(MultitaskTrainer, trainer).get_eval_dataloader(eval_dataset=eval_dataset[task_name])
        )
    preds_dict[task_name] = trainer.prediction_loop(
        eval_dataloader, 
        description=f"Validation: {task_name}"
    ) 

f = open(f"{exp_dir}/predictions.pkl","wb")
pickle.dump(preds_dict, f)
f.close()

stats_dict = {}
for task_name in dataset_dict:
    print(task_name)
    if task_name == "stsb":
        stats_dict[task_name] = datasets.load_metric('glue', 'stsb').compute(predictions = preds_dict["stsb"].predictions.flatten(), references = preds_dict["stsb"].label_ids)
    else:
        stats_dict[task_name] = datasets.load_metric('glue', task_name).compute(predictions = np.argmax(preds_dict[task_name].predictions, axis=1), references = preds_dict[task_name].label_ids)
    print(stats_dict[task_name])
    print()

with open(f"{exp_dir}/test_results.json", "w") as write_file:
    json.dump(stats_dict, write_file, indent=4)

os.remove(f"{exp_dir}/predictions.pkl")




