# EMNLP 2023: A State Vector Framework For Dataset Effects

This repository contains relevant scripts, notebooks, and results that supplement our EMNLP 2023 paper _A State Vector Framework for Dataset Effects_.

## Experiments

There were 33 different multitask GLUE dataset configurations used for model fine-tuning. Their ID and dataset composition can be accessed from the table below, with a checkmark indicating the inclusion of the task in the overall dataset.

| Experiment ID  | &nbsp; &nbsp; &nbsp; CoLA &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; SST-2 &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; MRPC &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; STSB &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; QNLI &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; RTE &nbsp; &nbsp; &nbsp; |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | ✔ |  |  |  |  |  |
| 2 |  | ✔ |  |  |  |  |
| 3 |  |  | ✔ |  |  |  |
| 4 |  |  |  | ✔ |  |  |
| 5 |  |  |  |  | ✔ |  |
| 6 |  |  |  |  |  | ✔ |
| 7 | ✔ | ✔ |  |  |  |  |
| 8 | ✔ |  | ✔ |  |  |  |
| 9 | ✔ |  |  | ✔ |  |  |
| 10 | ✔ |  |  |  | ✔ |  |
| 11 | ✔ |  |  |  |  | ✔ |
| 12 |  | ✔ | ✔ |  |  |  |
| 13 |  | ✔ |  | ✔ |  |  |
| 14 |  | ✔ |  |  | ✔ |  |
| 15 |  | ✔ |  |  |  | ✔ |
| 16 |  |  | ✔ | ✔ |  |  |
| 17 |  |  | ✔ |  | ✔ |  |
| 18 |  |  | ✔ |  |  | ✔ |
| 19 |  |  |  | ✔ | ✔ |  |
| 20 |  |  |  | ✔ |  | ✔ |
| 21 |  |  |  |  | ✔ | ✔ |
| 22 | ✔ |  | ✔ |  | ✔ |  |
| 23 | ✔ |  | ✔ |  |  | ✔ |
| 24 | ✔ |  |  | ✔ | ✔ |  |
| 25 | ✔ |  |  | ✔ |  | ✔ |
| 26 |  | ✔ | ✔ |  | ✔ |  |
| 27 |  | ✔ | ✔ |  |  | ✔ |
| 28 |  | ✔ |  | ✔ | ✔ |  |
| 29 |  | ✔ |  | ✔ |  | ✔ |
| 30 | ✔ | ✔ | ✔ | ✔ |  |  |
| 31 | ✔ | ✔ |  |  | ✔ | ✔ |
| 32 |  |  | ✔ | ✔ | ✔ | ✔ |
| 33 | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

## Python Scripts

- _multitask_nlp_train.py_: fine-tune baseline BERT/RoBERTa model
  - Command to run: `python3 multitask_nlp_train.py [experiment_id] [seed] [encoder] [model]`
    - `experiment_id`: integer from 1-33
    - `seed`: random seed options of 42, 1, 1234, 123, or 10
    - `encoder`: 'bert', 'roberta'
    - `model`: 'bert-base-cased', 'roberta-base'
- _shrink_probing_dataset.py_: reduce size of SentEval probing task datasets
  - Command to run: `python3 shrink_probing_datasets.py`
- _senteval_probing.py_: probe fine-tuned model on reduced SentEval probing datasets
  - Command to run: `python3 senteval_probing.py [experiment_id] [seed] [encoder] [model]`

## Notebooks

- _ind_effect_analysis.ipynb_: summary of individual effects of GLUE datasets on probing dimensions
- _visualize_dataset_effects.ipynb_: visualization of individual and interaction effects

## Results

In the `experiments` subfolder contains the probing and downstream task performances for each experiment. Note the ID '0' refers to the baseline encoder without any fine-tuning. 

- Path: `experiments/[encoder]/[seed]/[experiment_id]`
- Files:
  - _probe_results_: probing accuracies
  - _test_results_: downstream task performance metrics
