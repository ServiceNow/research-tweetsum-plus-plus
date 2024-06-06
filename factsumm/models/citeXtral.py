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
from transformers.trainer_callback import EarlyStoppingCallback
from factsumm.utils import concatenate_dialog
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import os
from copy import copy
import json
#from import Dataset
import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
# from factsumm.models.rouge_trainer import RougeTrainer
# from factsumm.metrics.rouge_metrics import RougeMetrics
import scipy
import evaluate
rouge = evaluate.load("rouge")

import wandb

# Set the _service_wait parameter to a higher value
wandb_config = {
    "_service_wait": 60.0  # Set timeout to 60 seconds
}

# Initialize wandb with the custom configuration
wandb.init(config=wandb_config)


class CiteXtral(MixtralSumm):
    def __init__(self):
        self.max_length = 2000 #2400  # verify if need to increase based on dataset
        self.base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" #"mistralai/Mixtral-8x7B-v0.1"  # "facebook/opt-125m"  # "mistralai/Mixtral-8x7B-v0.1"
        self.prompt_ending = "Extractive Summary and Abstractive Summary with References and Tags:"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tools = [TavilySearchResults(max_results=1)]
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.model_name = 'gpt-3.5-turbo-1106'
        self.llm = ChatOpenAI(model=self.model_name)
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True
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
        self.model_name = "citeXtral"
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

        # with open(
        #     "prompts/extsum_absum_prompt_2-shot_1.txt",
        #     "r",
        # ) as file:
        #     extractive_abstractive_prompt = file.read()

        
        with open(
            "prompts/abssum_w_dialogue_prompt_2-shot_1.txt",
            "r"
        ) as file:
            abstractive_w_dialog_prompt = file.read()
            
        self.abstractive_w_dialog_prompt = abstractive_w_dialog_prompt
        
        with open(
            # "prompts/extsum_absum_prompt_1-shot_1.txt",
            # "prompts/extsum_absum_prompt_no_few_shot.txt",
            "prompts/extsum_absum_prompt_no_few_shot_v2.txt",
            "r",
        ) as file:
            extractive_abstractive_prompt_no_few_shot = file.read()

        self.extractive_abstractive_prompt_no_few_shot = extractive_abstractive_prompt_no_few_shot
        self.extractive_abstractive_prompt = extractive_abstractive_prompt_no_few_shot #using this because it was trained this way
        

        # with open(  
        #     "prompts/assessment_prompt.txt",
        #     "r",
        # ) as file:
        #     assessment_prompt = file.read()
        # self.assessment_prompt = assessment_prompt
        
    def decode_predictions_and_labels(self, eval_pred):
        predictions, labels = eval_pred
        decoded_predictions = []
        decoded_labels = []
        
        # Loop through each prediction in the batch
        for prediction in predictions:
            decoded_string = self.tokenizer.decode(np.argmax(scipy.special.softmax(prediction, axis=-1), axis=-1))
            # Append the decoded string to the list of decoded predictions
            clean_decoded_string = decoded_string.split('vspace')[-1].strip()
            decoded_predictions.append(clean_decoded_string)  
        decoded_labels = self.tokenizer.batch_decode(np.where(labels != -100, labels, self.tokenizer.pad_token_id))  
        return decoded_predictions, decoded_labels
        
    def load_checkpoint(self, checkpoint_path):
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        
    # def invoke_assessment(self, self_labeled_example_batched):
    def invoke_assessment(self, self_labeled_example, dialogue):
        # add this - Give a point if the extractive summary serves as a justification for the abtractive summary faithfully represents the Extractive Summary
        criterion = """ Review the user's question and the corresponding response using the additive scoring system described below. 
        Points are accumulated based on the satisfaction of each criterion.
        - Add 1 point if the Extractive Summary contains only the most important sentences taken from the original dialogue. Give 0 points if more than 10 sentences are extracted.
        - For each sentence in the Abstractive Summary, add another point if it is supported by the referenced sentences of the Extractive Summary.
        - For each sentence in the Abstractive summary, award another point if it is abstracting the referenced sentences of the Extractive Summary and not just copying them. 
        - For each sentence in the Abstractive Summary, award another point if it the tags amongst ISSUE, FEEDBACK, QUESTION, RESOLUTION, RESOLUTION STEP, DEFLECTION, or WORKAROUND have been appropriately applied
        """
        #- For each sentence in the Abstractive Summary, award another point if it has tags such as ISSUE, FEEDBACK, QUESTION, RESOLUTION, RESOLUTION STEP, DEFLECTION, or WORKAROUND and they have been appropriately applied

        score_question = """After examining the user's instruction and the response:
        - Briefly justify your total score, up to 100 words.
        - Explicitly say when any point is not given and explain why.
        - Divide the total score by the maximum possible score to obtain a normalized score between 0 and 1.
        - Conclude with the normalized score using the format: 'Normalized Score: <normalized score>'
        To evaluate the response in alignment with this additive scoring model, we'll systematically attribute points based on the outlined criteria.
        """
        output_dir_confidence = os.path.join(self.self_labeled_path, f"self_assess_confidence_cycle_number_{self.cycle_number}")
        os.makedirs(output_dir_confidence, exist_ok=True)
        eval_prompt = (
            "Original dialogue" + "\n\n" + dialogue + "\n\n" +
            "User: " +
            criterion
            + " \n\n"
            + "<response>"
            + self_labeled_example
            + "</response>"
            + score_question
        )
        if self.assessment_model == "mixtral":  
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
                outputs = self.eval_tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                assessment_output = outputs[0]

        elif self.assessment_model == "gpt":  
            outputs = self.agent_executor.invoke({"input": eval_prompt})
            assessment_output = outputs['output']
        
        try:
            # ratio = float(outputs[0].strip().split('points')[-2].strip()[-1])
            fraction_str = assessment_output.split('Normalized Score:')[-1].strip().split('\n')[0]
            if len(fraction_str.split('/')) > 1:
                numerator, denominator = map(int, fraction_str.split('/'))
                ratio = float(numerator)/float(denominator)
            else:
                ratio = float(fraction_str)
        except:
            try:
                fraction_str = assessment_output.split("Normalized Score is")[-1].strip().split('\n')[0].strip('.')
                if "</normalize score>" in fraction_str:
                    fraction_str = fraction_str.split("</normalize score>")[0]
                if len(fraction_str.split('/')) > 1:
                    numerator, denominator = map(int, fraction_str.split('/'))
                    ratio = float(numerator)/float(denominator)
                else:
                    ratio = float(fraction_str)
            except:
                ratio=0
 
        return ratio, assessment_output
    
    def train(self):
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
        self.project = "tweetsum-finetune"
        self.base_model_name = "mixtral_extabs_prompt"
        self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name
        self.output_dir = (
            os.path.join(f"{self.base_path}/runs/", self.run_name))
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except PermissionError:
            self.output_dir = (
            os.path.join(f"{self.base_data_path_remote}/runs/", self.run_name)
        )
        print('output_dir ',self.output_dir)
        # load finetuned model to do predictions
        # if there is a checkpoint in the output_dir, load the one in the last forl
        if os.path.exists(self.output_dir):
            checkpoints = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint")]
            if len(checkpoints) > 0:
                self.checkpoint_path = os.path.join(
                    self.output_dir, checkpoints[-1])
                print("checkpoint_path ", self.checkpoint_path)
                self.model.load_adapter(self.checkpoint_path)
        else:
            self.model.load_adapter(self.checkpoint_path)
            
        # label with extsum_abssum finetune model

        if not self.skip_training:

            self.extractive_abstractive_prompt = self.extractive_abstractive_prompt_no_few_shot # use without few shot prompt because it was trained this way
            for cycle_number in range(self.num_cycles):
                self.cycle_number = cycle_number
                print('starting cycle number', cycle_number)
                examples_to_label = self.init_unlabeled_dataset()
                examples_to_label=examples_to_label[:int(self.unlabeled_examples_to_consider)] #use only 3 for testing pipeline
                output_dir = os.path.join(self.self_labeled_path, f"self_labeled_responses_cycle_number_{cycle_number}")
                os.makedirs(output_dir, exist_ok=True)
                self_labeled_responses=[]
                dialogues=[]
                # loop over examples to label
                for i in range(0, len(examples_to_label)):
                    example_to_label = examples_to_label[i]
                    dialog = concatenate_dialog(example_to_label)
                    # check if the file f"{i*}_response.json" exists
                    if os.path.exists(os.path.join(output_dir, f"{i}_response.json")):
                        with open(os.path.join(output_dir, f"{i}_response.json"), "r") as file:
                            self_labeled_responses.append(json.load(file))
                        if os.path.exists(os.path.join(output_dir, f"{i}_dialog.json")):
                            with open(os.path.join(output_dir, f"{i}_dialog.json"), "r") as file:
                                dialogues.append(json.load(file))
                    else:
                        self_labeled_response = self.invoke_extsum_abssum(dialog)
                        self_labeled_responses.append(self_labeled_response)
                    
                        id_response_path = os.path.join(output_dir, f"{i}_response.json")
                        dialogues.append(dialog)
                        with open(id_response_path, "w") as file:  # Save the response to a file
                            json.dump(self_labeled_response, file)
                        id_dialog_path = os.path.join(output_dir, f"{i}_dialog.json")
                        with open(id_dialog_path, "w") as file:
                            json.dump(dialog, file)    
                # load back original model and label with assessment model
                if self.assessment_model == "mixtral":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id, quantization_config=self.bnb_config, device_map="auto"
                    )
                self_labeled_confidences=[]
                self_labeled_assessments=[]
                self_labeled_dialogs = []
                for j, (dialogue, response) in enumerate(zip(dialogues, self_labeled_responses)):
                    # check if the file f"{i*}_assessment.json" exists
                    if os.path.exists(os.path.join(output_dir, f"{j}_assessment.json")):
                        with open(os.path.join(output_dir, f"{j}_assessment.json"), "r") as file:
                            self_labeled_assessments.append(json.load(file))
                        if os.path.exists(os.path.join(output_dir, f"{j}_confidence.json")):
                            with open(os.path.join(output_dir, f"{j}_confidence.json"), "r") as file:
                                self_labeled_confidences.append(json.load(file))
                        if os.path.exists(os.path.join(output_dir, f"{i}_dialog.json")):
                            with open(os.path.join(output_dir, f"{i}_dialog.json"), "r") as file:
                                self_labeled_dialogs.append(json.load(file))
                    else:
                        self_labeled_confidence, self_labeled_assessment = self.invoke_assessment(response, dialogue)
                        self_labeled_confidences.append(self_labeled_confidence)
                        self_labeled_dialogs.append(dialogue)
                        self_labeled_assessments.append(self_labeled_assessment)
                        id_assessment_path = os.path.join(output_dir, f"{j}_assessment.json")
                        with open(id_assessment_path, "w") as file:  # Save the response to a file
                            json.dump(self_labeled_assessment, file)
                        id_confidence_path = os.path.join(output_dir, f"{j}_confidence.json")
                        with open(id_confidence_path, "w") as file:
                            json.dump(self_labeled_confidence, file)   
                            
                print('self_labeled_responses', self_labeled_responses)
                top_self_labeled_responses = []
                top_self_labeled_dialogs = []
                top_idx = [k for k in range(len(self_labeled_confidences)) if self_labeled_confidences[k] > self.confidence_threshold]
                top_self_labeled_responses.extend(self_labeled_responses[m] for m in top_idx)
                top_self_labeled_dialogs.extend(self_labeled_dialogs[n] for n in top_idx)
                
                number_accepted_sl_file = os.path.join(output_dir, "number_accepted_sl.txt")
                with open(number_accepted_sl_file, "w") as file:
                    file.write(str(len(top_self_labeled_responses)))
                
                # # load back finetuned model and continue finetuning
                if self.assessment_model == "mixtral":
                    self.model.load_adapter(self.checkpoint_path)

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

                # Convert this list of dictionaries into a Dataset
                # convert top_self_labeled_responses and top_self_labeled_dialogs to dataset
                new_data = Dataset.from_dict(
                    {
                        "response": top_self_labeled_responses,
                        "dialogue": top_self_labeled_dialogs,
                    }
                )
                self.tokenized_train_dataset2 = new_data.map(
                    self.generate_and_tokenize_training_prompt_for_unlabeled
                )
                self.init_train_dataset()
                self.combined_train_dataset = concatenate_datasets(
                    [self.tokenized_train_dataset, self.tokenized_train_dataset2]
                )
                def compute_metrics(eval_pred):
                    decoded_preds, decode_labels = self.decode_predictions_and_labels(eval_pred)
                    result = rouge.compute(predictions=decoded_preds, references=decode_labels)
                    predictions, _ = eval_pred
                    prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
                    result["gen_len"] = np.mean(prediction_lens)
                    return {k: round(v, 4) for k, v in result.items()}

                # self.trainer = RougeTrainer(
                self.trainer = transformers.Trainer(   
                    model=self.model,
                    train_dataset=self.combined_train_dataset,
                    eval_dataset=self.tokenized_val_dataset,
                    tokenizer = self.tokenizer,
                    compute_metrics=compute_metrics, # try it!
                    args=transformers.TrainingArguments(
                        output_dir=self.output_dir,
                        warmup_steps=5,
                        per_device_train_batch_size=int(self.per_device_train_batch_size),
                        gradient_checkpointing=True,
                        gradient_accumulation_steps=4,
                        max_steps=self.max_steps, #1000,  # 100,  # 50, #500,
                        learning_rate=float(self.learning_rate),
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
                early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)
                self.trainer.add_callback(early_stopping_callback)
                # self.trainer.add_callback(transformers.EvalPredictionRougeCallback())
                self.model.config.use_cache = False

                self.trainer.train()
       
        
