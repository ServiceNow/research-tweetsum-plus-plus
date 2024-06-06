# TweetSumm++

This is repository for the paper TweetSumm++ for Evaluating and Enhancing Factuality and Coverage in Abstractive Summarization that References Extracted Content


## Running experiments

### 1. Clone the repo and setup up a conda environment with python>=3.9

Below instructions are for conda/miniconda but you can also use virtualenv

```
git clone git@github.com:ServiceNow/research-tweetsum-plus-plus.git
cd research-tweetsum-plus-plus

cconda create -n factsumm python=3.10 -y
conda activate factsumm
pip install -e .
```


### 2. Get the Datasets

Download the datasets from: https://huggingface.co/datasets/gebelangsn/tweetsum_plus_plus
Use the password in the paper's Appendix to open the password-protected zip files.
Copy it in the base directory of this repository.

### 3. Run different models and datasets on Tweetsum++
Change line 251, 252, and 253 of factsum/summarizer_test.py with the values of the model name, dataset name and config name you want to run respectively.


Chosse the model <model_name> you want to run among the following: 
Baselines: "random", "lead", "presumm_extractive", 
Mixtral SFT: "mixtralsumm"
Mixtral SFT + Self-Labeling: "citeXtral"
GPT3.5: "gpt-3.5-turbo-1106"
GPT4: "gpt-4"
GPT3.5 abstractive-only: "gpt-3.5-turbo-1106/abstractive"
GPT4 abstractive-only: "gpt-4/abstractive"

Choose the dataset <dataset_name> you want to run among the following:
Original tweetsum: "tweetsum"
Tweetsum++: "tweetsum++_v1"

Choose the file name <file_name> with right configuration: 
Random: "configs/random.yaml"
Lead "configs/lead.yaml"
Mixtal-instruct 1-shot: configs/summarizer_mixtral_instruct_1-shot_110_examples.yaml 
Mixtral-SFT: configs/summarizer_mixtral_instruct_train_301_example_test_110_example_max_steps_1000_no_few_shot_lr_2e-5.yaml 
Mixtral-SFT + SL: configs/summarizer_citeXtral_train_301_max_new_tokens_450_4_unlabeled_cycles_1_max_steps_1000_batch_1_score_mixtral_instruct_train_assess_normalized_only_1scores_rouge_val.yaml
GPT3.5 1-shot: configs/summarizer_gpt3_110_examples.yaml
GPT4 1-shot: configs/summarizer_gpt4_110_examples.yaml

Run the following command to run the Tweetsum++ pipeline

```bash
$ python factsum/summarizer_test.py 
```

### 4. Results
An output directory ====Output directory==== <output_directory> will be printed. There you can find the results of the run.
