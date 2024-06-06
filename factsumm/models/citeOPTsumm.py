import transformers
from datetime import datetime
from factsumm.models.mixtralsumm import MixtralSumm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import os
from copy import copy
import json
from factsumm.models.citeXtral import CiteXtral

class CiteOptSumm(CiteXtral):
    def __init__(self):
        self.max_length = 2400  # verify if need to increase based on dataset
        self.base_model_id = "facebook/opt-125m"  # "mistralai/Mixtral-8x7B-v0.1"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, quantization_config=self.bnb_config, device_map="auto"
        )
        # self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, add_eos_token=True, add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, add_bos_token=True, trust_remote_code=True
        )
        self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
        self.model_name = "citeOPT"
        # self.init_dataset()
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add padding token

        with open(  # TODO: EXPERIMENT WITH ADDING FEW SHOT PROMPT TO EXTSUM
            "prompts/extsum_prompt_2-shot_1.txt",
            "r",
        ) as file:
            extractive_prompt = file.read()
        self.extractive_prompt = extractive_prompt

        with open(
            "prompts/abssum_prompt_2-shot_1.txt",
            "r",
        ) as file:
            abstractive_prompt = file.read()

        self.abstractive_prompt = abstractive_prompt

        with open(
            "prompts/extsum_absum_prompt_2-shot_1.txt",
            "r",
        ) as file:
            extractive_abstractive_prompt = file.read()

        self.extractive_abstractive_prompt = extractive_abstractive_prompt

        # with open(  # TODO: EXPERIMENT WITH ADDING FEW SHOT PROMPT TO EXTSUM
        #     "prompts/ai_feedback_prompt.txt",
        #     "r",
        # ) as file:
        #     ai_feedback_prompt = file.read()
        # self.ai_feedback_prompt = ai_feedback_prompt
        
    def load_checkpoint(self, checkpoint_path):
        self.model = self.model # avoid loading just to debub# PeftModel.from_pretrained(self.model, checkpoint_path)
    
    # def train(self):
    #     #copy self.model to self.original_model and then load checkpoint to self.model
    #     self.assesment_model = copy(self.model)
        
    #     self.model = self.load_checkpoint(self.checkpoint_path)
        
    #     self.init_dataset()
        
    #     self_labeled_dialogs = []
    #     #predict using assesment_model on concatenated_unlabeled_dialogs
    #     for unlabeled_dialog in self.concatenated_unlabeled_dialogs:
    #         self_labeled_dialog=self.invoke_extsum_abssum(unlabeled_dialog)
    #         self_labeled_dialogs.append(self_labeled_dialog)
            
    #     #tokenize self_labeled_dialogs
    #     self_labeled_dialogs = self.tokenizer(self_labeled_dialogs, return_tensors="pt", padding=True, truncation=True)

    #     self.tokenized_train_dataset = self.tokenized_train_dataset + self_labeled_dialogs
            
            
    #     if self.local_run:
    #         self.base_path = self.base_data_path_local
    #     else:
    #         self.base_path = self.base_data_path_remote
    #     # prepare_model_for_kbit_training(self.model)
    #     self.model.gradient_checkpointing_enable()
    #     self.model = prepare_model_for_kbit_training(self.model)
    #     self.config = LoraConfig(
    #         r=8,
    #         lora_alpha=16,
    #         target_modules=[
    #             "q_proj",
    #             "k_proj",
    #             "v_proj",
    #             "out_proj",
    #             "w1",
    #             "w2",
    #             "w3",
    #             "lm_head",
    #         ],
    #         bias="none",
    #         lora_dropout=0.05,
    #         task_type="CAUSAL_LM",
    #     )

    #     self.model = get_peft_model(self.model, self.config)

    #     # self.model = self.accelerator.prepare(self.model)

    #     if torch.cuda.device_count() > 1:
    #         self.model.is_parallelizable = True  # = torch.nn.DataParallel(self.model)
    #         self.model.model_parallel = True
    #     print("torch.cuda.device_count()", torch.cuda.device_count())

    #     self.project = "tweetsum-finetune"
    #     self.base_model_name = "mixtral_extabs_prompt"
    #     self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name
    #     self.output_dir = (
    #         os.path.join(f"{self.base_path}/runs/", self.run_name)
    #     )
    #     #wandb.init(project=self.project, name=self.run_name)

    #     self.tokenizer.pad_token = self.tokenizer.eos_token

    #     self.trainer = transformers.Trainer(
    #         model=self.model,
    #         train_dataset=self.tokenized_train_dataset,
    #         eval_dataset=self.tokenized_val_dataset,
    #         args=transformers.TrainingArguments(
    #             output_dir=self.output_dir,
    #             warmup_steps=5,
    #             per_device_train_batch_size=1,
    #             gradient_checkpointing=True,
    #             gradient_accumulation_steps=4,
    #             max_steps=self.max_steps, #1000,  # 100,  # 50, #500,
    #             learning_rate=2.5e-5,
    #             logging_steps= self.logging_steps, #5,  # 25,
    #             fp16=True,
    #             optim="paged_adamw_8bit",
    #             logging_dir="./logs",
    #             save_strategy="steps",
    #             save_steps=self.save_steps, #50,  # 10,  # 50,
    #             eval_steps=self.eval_steps, #50,  # 10,  # 50,
    #             do_eval=True,
    #             report_to="wandb",
    #             run_name=f"{self.run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    #             evaluation_strategy="steps",
    #         ),
    #         data_collator=DataCollatorForLanguageModeling(
    #             tokenizer=self.tokenizer, mlm=False
    #         ),
    #     )
    #     self.model.config.use_cache = False
    #     if not self.skip_training:
    #         self.trainer.train()
