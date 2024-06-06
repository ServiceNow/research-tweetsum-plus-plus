# from TransformerSum.src.extractive import ExtractiveSummarizer
from factsumm.models.extractive import ExtractiveSummarizer
from factsumm.models.abstractive import AbstractiveSummarizer
from pytorch_lightning import Trainer
import argparse
import os
from factsumm.models.backbone import Backbone
import pandas as pd
from factsumm.metrics.rouge_eval_extractive import RougeEvalExtractive
from factsumm.utils import (
    handle_extractive_predictions,
)


class PreSumm(Backbone):
    def __init__(self, args):
        # self.model = ExtractiveSummarizer.load_from_checkpoint()
        self.max_length = 514  # verify if need to increase based on dataset
        self.extractive_model = ExtractiveSummarizer.load_from_checkpoint(
            # "/mnt/colab_public/gebelangsn/pretrained_checkpoints/bert-base-uncased-ext-sum/epoch3.ckpt",
            "/mnt/colab_public/gebelangsn/pretrained_checkpoints/distilroberta-base-ext-sum/epoch3.ckpt",
            # "/home/toolkit/models/bert-base-uncased/epoch3.ckpt",
            strict=False,
            # hparams_file="configs/summarizer_presumm_extractive_2_examples.yaml",
        )
        self.abstractive_model = AbstractiveSummarizer(args)
        # .load_from_checkpoint(
        #     "/mnt/colab_public/gebelangsn/pretrained_checkpoints/distil-led-large-16384/tokenizer_config.json",
        #     strict=False,
        # )
        self.tokenizer = self.extractive_model.tokenizer
        self.abstractive_tokenizer = self.abstractive_model.tokenizer
        self.eval_tokenizer = self.tokenizer
        args["amp_backend"] = "apex"
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.trainer = Trainer.from_argparse_args(args)
        
        
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
        

    def train(self):
        self.init_dataset()
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
        self.project = "tweetsum-finetune"
        self.base_model_name = "presumm_extabs_prompt"
        # self.run_name = self.base_model_name + "-" + self.project
        self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name
        # self.output_dir = "./" + self.run_name
        self.output_dir = (
            # "/mnt/colab_public/data/gebelangsn/factsumm/checkpoints/" + self.run_name
            # "/mnt/colab_public/data/gebelangsn/factsumm/runs/" + self.run_name
            os.path.join("/mnt/colab_public/data/gebelangsn/factsumm/runs/", self.run_name)
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.skip_training:
            self.trainer.fit(self.extractive_model)
        

    def test(self, text):
        return self.trainer.test(self.extractive_model)

    def init_dataset(self):
        self.tokenized_train_dataset = self.dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_prompt2
        )
        self.tokenized_val_dataset = self.dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_prompt2
        )
        
    def tokenize(self, prompt):
        # result = self.tokenizer(prompt)
        result = self.extractive_model.tokenizer(
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
    
    def invoke_extsum(self, dialog):
        # tokenized_dialog = self.tokenize(dialog)
        # output = self.model(input_ids = tokenized_dialog['input_ids'],
        #                     attention_mask = tokenized_dialog['attention_mask'])
                            # **tokenized_dialog)
        output = self.extractive_model.predict(dialog)
        return output        
    
    def invoke_abssum_w_extsum(self, pred_extsum):
        # tokenized_pred_extsum = self.tokenize(pred_extsum)
        # tokenized_pred_extsum = self.abstractive_tokenizer(pred_extsum)
        pred_extsum_input_ids = self.abstractive_tokenizer.encode(pred_extsum, return_tensors="pt")
        
        output = self.abstractive_model(source = pred_extsum_input_ids)
        decoded_output = self.abstractive_tokenizer.decode(output[0], skip_special_tokens=True)
        
        return decoded_output
    
    def predict(self):
            extsum_responses = []
            extsum_summaries = []
            for i, dialog in enumerate(self.dataset["dialog_formatted"]):
                extsum_response = self.invoke_extsum(dialog)
                extsum_responses.append(extsum_response)
                if "\n\n" in extsum_response:
                    extsum_summaries.append(extsum_response.split("\n\n")[1])
                else:
                    extsum_summaries.append(extsum_response)

            self.predicted_extractive_summaries = extsum_summaries
            if type(self.dataset) == pd.DataFrame:
                self.pandas_dataset = self.dataset
            else:
                self.pandas_dataset = self.dataset.to_pandas()
            #TODO: first parse the references and then clean the extractive summaries from the line numbers
            self.pandas_dataset['raw_predicted_extractive_summaries'] = self.predicted_extractive_summaries
            self.pandas_dataset['handled_predicted_extractive_summaries'] = handle_extractive_predictions(self.pandas_dataset)
            

            # save dataframe
            self.pandas_dataset.to_csv(
                # f"/mnt/colab_public/data/gebelangsn/factsumm/outputs/predicted_summaries_{self.model_name}_{timestamp}.csv",
                os.path.join(self.output_dir, "predicted_summaries.csv"),
                index=False,
            )
            
    def evaluate(self):
        if self.do_predict:
            self.predict()
        else:
            pass # un-enabling loading predictions for now
   

        rouge_evaluator_extractive = RougeEvalExtractive()
        (
            avg_rouge1_extractive,
            avg_rouge2_extractive,
            avg_rougeL_extractive,
            # avg_rouge1_abstractive,
            # avg_rouge2_abstractive,
            # avg_rougeL_abstractive,
            avg_rouge1_extsum_on_ext_gt,
            avg_rouge2_extsum_on_ext_gt,
            avg_rougeL_extsum_on_ext_gt,
        ) = rouge_evaluator_extractive.evaluate(self.pandas_dataset)

        # save average scores to a file with timestamp and model name
        with open(
            os.path.join(self.output_dir, "ave_rouge_scores.txt"),
            "w",
        ) as file:
            file.write(
                f"avg_rouge1_extractive: {avg_rouge1_extractive}\n"
                f"avg_rouge2_extractive: {avg_rouge2_extractive}\n"
                f"avg_rougeL_extractive: {avg_rougeL_extractive}\n"
                # f"avg_rouge1_abstractive: {avg_rouge1_abstractive}\n"
                # f"avg_rouge2_abstractive: {avg_rouge2_abstractive}\n"
                # f"avg_rougeL_abstractive: {avg_rougeL_abstractive}\n"
                f"avg_rouge1_extsum_on_ext_gt: {avg_rouge1_extsum_on_ext_gt}\n"
                f"avg_rouge2_extsum_on_ext_gt: {avg_rouge2_extsum_on_ext_gt}\n"
                f"avg_rougeL_extsum_on_ext_gt: {avg_rougeL_extsum_on_ext_gt}\n"
            )