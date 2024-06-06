import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
import matplotlib.pyplot as plt
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import transformers
from datetime import datetime
import os
from factsumm.models.backbone import Backbone
from transformers.trainer_callback import EarlyStoppingCallback


class MixtralSumm(Backbone):
    def __init__(self):
        self.max_length = 2000  # verify if need to increase based on dataset
        self.base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" #"mistralai/Mixtral-8x7B-v0.1"  # "facebook/opt-125m"  # "mistralai/Mixtral-8x7B-v0.1"
        self.prompt_ending = "Extractive Summary and Abstractive Summary with References and Tags:"
        self.abssum_prompt_ending = "Abstractive Summary with References and Tags:"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # self.device = torch.device("cpu")
        # self.model  = None
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
        self.model_name = "mixtral"
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
            # "prompts/extsum_absum_prompt_1-shot_1.txt",
            "prompts/extsum_absum_prompt_2-shot_1.txt",
            "r",
        ) as file:
            extractive_abstractive_prompt = file.read()

        self.extractive_abstractive_prompt = extractive_abstractive_prompt
        
        with open(
            # "prompts/extsum_absum_prompt_1-shot_1.txt",
            "prompts/extsum_absum_prompt_no_few_shot.txt",
            "r",
        ) as file:
            extractive_abstractive_prompt_no_few_shot = file.read()

        self.extractive_abstractive_prompt_no_few_shot = extractive_abstractive_prompt_no_few_shot
        
        with open(
            "prompts/abssum_w_dialogue_prompt_2-shot_1.txt",
            "r"
        ) as file:
            abstractive_w_dialog_prompt = file.read()
            
        self.abstractive_w_dialog_prompt = abstractive_w_dialog_prompt


    def tokenize(self, prompt):
        # result = self.tokenizer(prompt)
        result = self.tokenizer(
            prompt,
            return_tensors="pt",  # None,  # "pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        result["input_ids"] = result["input_ids"].squeeze(0)
        result["attention_mask"] = result["attention_mask"].squeeze(0)
        # result["labels"] = result["input_ids"].copy()
        result["labels"] = result["input_ids"].clone()
        return result

    def summarize(self):
        return "Mixtral Summarization"

    def plot_data_lengths(self, tokenized_train_dataset, tokenized_val_dataset):
        lengths = [len(x["input_ids"]) for x in tokenized_train_dataset]
        lengths += [len(x["input_ids"]) for x in tokenized_val_dataset]
        print(len(lengths))

        # plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7, color="blue")
        plt.xlabel("length of input_ids")
        plt.ylabel("frequency")
        plt.title("distribution of input_ids length")
        plt.show()

    def calculate_max_length(self):
        self.init_dataset()
        self.plot_data_lengths(self.tokenized_train_dataset, self.tokenized_val_dataset)

    def train(self, skip_training=False, local_run=False, checkpoint_path=None, save_dir=None, max_steps=1000, logging_steps=5, save_steps=50, eval_steps=50, gradient_accumulation_steps=1, per_device_train_batch_size=1, per_device_eval_batch_size=1):
        if not hasattr(self, "per_device_train_batch_size"):
            self.per_device_train_batch_size = per_device_train_batch_size
        if not hasattr(self, "per_device_eval_batch_size"):
            self.per_device_eval_batch_size = per_device_eval_batch_size
        if not hasattr(self, "max_steps"):
            self.max_steps = max_steps
        if not hasattr(self, "gradient_accumulation_steps"):
            self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = self.model_max_length
        if self.skip_training:
            self.init_dataset()
        else:
            self.init_train_dataset()
            # self.init_train_dataset_two_steps()
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
        # prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "w1",
                "w2",
                "w3",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, self.config)

        # self.model = self.accelerator.prepare(self.model)

        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True  # = torch.nn.DataParallel(self.model)
            self.model.model_parallel = True
        print("torch.cuda.device_count()", torch.cuda.device_count())

        self.project = "tweetsum-finetune"
        self.base_model_name = f"mixtral_extabs_prompt-lr-{self.learning_rate}"
        self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name + "-" + datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.output_dir = (
            os.path.join(f"{self.base_path}/runs/", self.run_name)
        )
        #wandb.init(project=self.project, name=self.run_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=self.output_dir,
                warmup_steps=5,
                per_device_train_batch_size=self.per_device_train_batch_size,
                gradient_checkpointing=True,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                max_steps=self.max_steps, #1000,  # 100,  # 50, #500,
                learning_rate=float(self.learning_rate),  # 5e-5,  # 1e-4,
                logging_steps= self.logging_steps, #5,  # 25,
                fp16=True,
                optim="paged_adamw_8bit",
                logging_dir="./logs",
                save_strategy="steps",
                save_steps=self.save_steps, #50,  # 10,  # 50,
                eval_steps=self.eval_steps, #50,  # 10,  # 50,
                do_eval=True,
                report_to="wandb",
                load_best_model_at_end=True,
                run_name=f"{self.run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                evaluation_strategy="steps",
            ),
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
        self.trainer.add_callback(early_stopping_callback)
        self.model.config.use_cache = False
        if not self.skip_training:
            self.trainer.train()

    def load_ft_model(self):
        self.ft_model = PeftModel.from_pretrained(
            self.model,
            self.checkpoint_path,  # "mixtral-tweetsum-finetune/checkpoint-500"
        )

    # def invoke_extsum(self, dialog):
    #     eval_prompt = (
    #         self.extractive_prompt + " \n\n" + concatenate_dialog(dialog) + "\n\n" + "Extractive Summary 2:"
    #     )

    #     model_input = self.eval_tokenizer(
    #         eval_prompt,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_length,
    #     )
    #     model_input = model_input.to("cuda")

    #     self.model.eval()
    #     with torch.no_grad():

    #         output = self.eval_tokenizer.decode(
    #             self.model.generate(**model_input, max_new_tokens=200)[0],
    #             skip_special_tokens=True,
    #         )
    #         print(output)
    #         extsum_output = output.split("Extractive Summary 2:")[1].strip()

    #         return extsum_output

    def invoke_extsum_abssum(self, dialogue):            
        eval_prompt = (
            self.extractive_abstractive_prompt
            + " \n\n"
            + dialogue
            + "\n\n"
            + self.prompt_ending
        )
            # eval_prompts.append(eval_prompt)
        model_input = self.eval_tokenizer(
            eval_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        model_input = model_input.to("cuda")

        self.model.eval()
        with torch.no_grad():
            
            output = self.model.generate(
                **model_input,
                max_new_tokens=self.max_new_tokens,
            )
            decoded_output = self.eval_tokenizer.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            print(decoded_output)
            try:
                extsum_abssum_output = decoded_output[0].split(
                    self.prompt_ending
                )[-1].strip()
                # extsum_abssum_outputs.append(extsum_abssum_output)
            except:
                extsum_abssum_output = decoded_output[0].strip()
                # extsum_abssum_outputs.append(extsum_abssum_output)
            return extsum_abssum_output

    def invoke_abssum_w_extsum(self, pred_extsum):
        eval_prompt = (
            self.abstractive_prompt
            + " \n\n"
            + pred_extsum
            + "\n\n"
            + self.abssum_prompt_ending
        )
        model_input = self.eval_tokenizer(
            eval_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        model_input = model_input.to("cuda")

        self.model.eval()
        with torch.no_grad():

            output = self.eval_tokenizer.decode(
                self.model.generate(**model_input, max_new_tokens=self.max_new_tokens)[0],
                skip_special_tokens=True,
            )
            print(output)
            try:
                abssum_output = output.split(
                    self.abssum_prompt_ending
                )[1].strip()
            except:
                abssum_output = output.strip()
            return abssum_output

    def load_checkpoint(self, checkpoint_path):
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
