from factsumm.utils import (
    handle_extractive_predictions,
    handle_abstractive_predictions,
    handle_reference_predictions,
    handle_tag_predictions,
    handle_reference_predictions_fuzzy_match,
    handle_reference_gt_fuzzy_match,
    concatenate_dialog
)
import pandas as pd
import os
from factsumm.metrics.rouge_eval import RougeEval
from factsumm.metrics.prf1_eval import PRF1
from factsumm.metrics.similarity_eval import SimilarityEval
import torch
import json
import random

class Backbone:
    def evaluate(self):
        if self.do_predict:
            self.predict()
        else:
            pass # un-enabling loading predictions for now
            # load csv in gpt3_predictions_path into pandas dataframe
            # self.pred_df = pd.read_csv(self.predictions_path)

            # self.predicted_abstractive_summaries = handle_abstractive_predictions(
            #     self.pred_df
            # )

            # self.predicted_extractive_summaries = handle_extractive_predictions(
            #     self.pred_df
            # )
        # if type(self.pred_df) == pd.DataFrame:
        #     self.pandas_dataset = self.pred_df
        # else:
        #     self.pandas_dataset = self.pred_df.to_pandas()
        # self.predicted_tags = handle_tag_predictions(self.pandas_dataset)
        # self.pandas_dataset["predicted_tags"] = self.predicted_tags
        self.pandas_dataset["predicted_tags"] = handle_tag_predictions(self.pandas_dataset)

        # self.predicted_line_references = handle_reference_predictions(
        #     self.pandas_dataset
        # )

        # self.pandas_dataset["predicted_line_references"] = (
        #     self.predicted_line_references
        # )
        # self.pandas_dataset['predicted_line_references'] = handle_reference_predictions(self.pandas_dataset)
        # self.concatenate_dialog(self.pandas_dataset['dialog_formatted'])
        self.pandas_dataset['dialog_formatted'] = self.pandas_dataset['dialog_formatted'].apply(concatenate_dialog)
        if self.dataset != 'tweetsum':
            self.pandas_dataset['predicted_line_references'] = handle_reference_predictions_fuzzy_match(self.pandas_dataset)
            self.pandas_dataset['ground_truth_line_references'] = handle_reference_gt_fuzzy_match(self.pandas_dataset)

        rouge_evaluator = RougeEval()
        (
            avg_rouge1_extractive,
            avg_rouge2_extractive,
            avg_rougeL_extractive,
            avg_rouge1_abstractive,
            avg_rouge2_abstractive,
            avg_rougeL_abstractive,
            avg_rouge1_extsum_on_ext_gt,
            avg_rouge2_extsum_on_ext_gt,
            avg_rougeL_extsum_on_ext_gt,
        ) = rouge_evaluator.evaluate(self.pandas_dataset)

        # save average scores to a file with timestamp and model name
        with open(
            os.path.join(self.output_dir, "ave_rouge_scores.txt"),
            "w",
        ) as file:
            file.write(
                f"avg_rouge1_extractive: {avg_rouge1_extractive}\n"
                f"avg_rouge2_extractive: {avg_rouge2_extractive}\n"
                f"avg_rougeL_extractive: {avg_rougeL_extractive}\n"
                f"avg_rouge1_abstractive: {avg_rouge1_abstractive}\n"
                f"avg_rouge2_abstractive: {avg_rouge2_abstractive}\n"
                f"avg_rougeL_abstractive: {avg_rougeL_abstractive}\n"
                f"avg_rouge1_extsum_on_ext_gt: {avg_rouge1_extsum_on_ext_gt}\n"
                f"avg_rouge2_extsum_on_ext_gt: {avg_rouge2_extsum_on_ext_gt}\n"
                f"avg_rougeL_extsum_on_ext_gt: {avg_rougeL_extsum_on_ext_gt}\n"
            )
        if self.dataset != 'tweetsum':
            prf1_evaluator = PRF1()
            average_random_precision, average_random_recall, average_random_f1 = (
                prf1_evaluator.evaluate(self.pandas_dataset)
            )
            # save average cores to a file with time stamp and model name
            with open(
                os.path.join(self.output_dir, "avg_prf1_scores.txt"),
                "w",
            ) as file:
                file.write(
                    f"average_random_precision: {average_random_precision}\n"
                    f"average_random_recall: {average_random_recall}\n"
                    f"average_random_f1: {average_random_f1}\n"
                )

            (
                average_tag_f1_RESOLUTION,
                average_tag_f1_RESOLUTION_STEP,
                average_tag_f1_WORKAROUND,
                average_tag_f1_ISSUE,
                average_tag_f1_DEFLECTION,
                average_tag_f1_FEEDBACK,
                average_tag_f1_QUESTION,
                average_f1_overall
            ) = prf1_evaluator.evaluate_per_tag(self.pandas_dataset)
            # save average cores to a file with time stamp and model name
            with open(
                os.path.join(self.output_dir, "avg_prf1_per_tag_scores.txt"),
                "w",
            ) as file:
                file.write(
                    f"average_tag_f1_RESOLUTION: {average_tag_f1_RESOLUTION}\n"
                    f"average_tag_f1_RESOLUTION_STEP: {average_tag_f1_RESOLUTION_STEP}\n"
                    f"average_tag_f1_WORKAROUND: {average_tag_f1_WORKAROUND}\n"
                    f"average_tag_f1_ISSUE: {average_tag_f1_ISSUE}\n"
                    f"average_tag_f1_DEFLECTION: {average_tag_f1_DEFLECTION}\n"
                    f"average_tag_f1_FEEDBACK: {average_tag_f1_FEEDBACK}\n"
                    f"average_tag_f1_QUESTION: {average_tag_f1_QUESTION}\n"
                    f"average_f1_overall: {average_f1_overall}\n"
                )
                
            similarity_evaluator = SimilarityEval()
            average_precision, average_recall, average_f1, average_ref_per_row =similarity_evaluator.evaluate(self.pandas_dataset)
            
            #write to file
            with open(
                os.path.join(self.output_dir, "avg_similarity_scores.txt"),
                "w",
            ) as file:
                file.write(
                    f"average_precision: {average_precision}\n"
                    f"average_recall: {average_recall}\n"
                    f"average_f1: {average_f1}\n"
                    f"average_ref_per_row: {average_ref_per_row}\n"
                )

    def predict(self):
            extsum_responses = []
            abssum_responses = []
            extsum_summaries = []
            abssum_summaries = []
            for i, dialog in enumerate(self.test_dataset["dialog_formatted"]):
                dialog_concatenated = concatenate_dialog(dialog)

                if self.extsum_absum_1_call:
                    extsum_abssum_response = self.invoke_extsum_abssum(dialog_concatenated)
                    extsum_responses.append(extsum_abssum_response)
                    print(extsum_abssum_response)
                    if '\n\n' in extsum_abssum_response:
                        list_response = extsum_abssum_response.split('\n\n')
                        if len(list_response[0]) > 10:
                            self.predicted_extractive_summaries = extsum_summaries.append(list_response[0]) #add handle_abstractive_predictions() add handle_extractive_predictions()
                            self.predicted_abstractive_summaries = abssum_summaries.append(list_response[1]) #TODO if predictions improve, then clean abssum
                        else:
                            self.predicted_extractive_summaries = extsum_summaries.append(list_response[1])
                            if len(list_response) < 2: # for the case where the abstractive summary is not generated
                                self.predicted_abstractive_summaries = abssum_summaries.append(list_response[1])
                            else:
                                self.predicted_abstractive_summaries = abssum_summaries.append(list_response[2])
                    else:
                        self.predicted_extractive_summaries =extsum_abssum_response
                        self.predicted_abstractive_summaries =extsum_abssum_response
                    
                else:
                    # extsum_response = self.model.predict(dialog)
                    extsum_response = self.invoke_extsum(dialog_concatenated)
                    extsum_responses.append(extsum_response)
                    if self.oracle_extsum:
                        abssum_response = self.invoke_absum_w_oracle_extsum(concatenate_dialog(self.test_dataset['extractive_summary'][i]))  
                    else:
                        # abssum_response = self.invoke_abssum_w_extsum(extsum_response)
                        abssum_response = self.invoke_abssum_w_dialogue_and_extsum(dialog_concatenated, extsum_response)
                        
                    abssum_responses.append(abssum_response)
                    # if "\n\n" in extsum_response:
                    #     extsum_summaries.append(extsum_response.split("\n\n")[1])
                    # else:
                    extsum_summaries.append(extsum_response)
                    # if "Abstractive Summary with References and Tags:\n" in abssum_response:
                    #     abssum_summaries.append(
                    #         abssum_response.split(
                    #             "Abstractive Summary with References and Tags:\n"
                    #         )[1]
                    #     )
                    # else:
                    abssum_summaries.append(abssum_response)
            self.predicted_extractive_summaries = extsum_summaries
            self.predicted_abstractive_summaries = abssum_summaries
            if type(self.test_dataset) == pd.DataFrame:
                self.pandas_dataset = self.test_dataset
            else:
                self.pandas_dataset = self.test_dataset.to_pandas()

            #TODO: first parse the references and then clean the extractive summaries from the line numbers
            self.pandas_dataset['raw_predicted_extractive_summaries'] = self.predicted_extractive_summaries
            self.pandas_dataset['raw_predicted_abstractive_summaries'] = self.predicted_abstractive_summaries
            self.pandas_dataset['predicted_line_references'] = handle_reference_predictions(self.pandas_dataset)
            self.pandas_dataset['predicted_tags'] = handle_tag_predictions(self.pandas_dataset)
            self.pandas_dataset['handled_predicted_extractive_summaries'] = handle_extractive_predictions(self.pandas_dataset)
            self.pandas_dataset['handled_predicted_abstractive_summaries'] = handle_abstractive_predictions(self.pandas_dataset)
            

            # save dataframe
            self.pandas_dataset.to_csv(
                # f"/mnt/colab_public/data/gebelangsn/factsumm/outputs/predicted_summaries_{self.model_name}_{timestamp}.csv",
                os.path.join(self.output_dir, "predicted_summaries.csv"),
                index=False,
            )
    def init_dataset(self):
        if self.dataset != 'tweetsum':
            self.tokenized_train_dataset = self.train_dataset.map(
                # self.generate_and_tokenize_prompt1
                self.generate_and_tokenize_prompt2
            )
            self.tokenized_val_dataset = self.test_dataset.map(
                # self.generate_and_tokenize_prompt1
                self.generate_and_tokenize_prompt2
            )
        else:
            self.tokenized_train_dataset = None
            self.tokenized_val_dataset = None
        
    def init_train_dataset(self):
        # split the train dataset into train and validation parts
        self.split_train_dataset = self.train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
        
        self.tokenized_train_dataset = self.split_train_dataset['train'].map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt
        )
        
        self.tokenized_val_dataset = self.split_train_dataset['test'].map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt
        )
    def init_train_dataset_two_steps(self):
        self.tokenized_train_dataset1 = self.train_dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt_extsum
        )
        self.tokenized_train_dataset2 = self.train_dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt_abssum
        )
        self.tokenized_test_dataset1 = self.test_dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt_extsum
        )
        self.tokenized_test_dataset2 = self.test_dataset.map(
            # self.generate_and_tokenize_prompt1
            self.generate_and_tokenize_training_prompt_abssum
        )
        #combine the two datasets # TODO: TEST!! AND SEND JOB
        self.tokenized_train_dataset = self.tokenized_train_dataset1 + self.tokenized_train_dataset2
        self.tokenized_val_dataset = self.tokenized_test_dataset1 + self.tokenized_test_dataset2
        
    def init_unlabeled_dataset(self):
        with open('list_of_unlabeled_turns.json', 'r') as file:
            list_of_unlabeled_turns = json.load(file)
        # shuffle the list
        random.shuffle(list_of_unlabeled_turns)
        # self.unlabeled_dataset = list_of_unlabeled_turns
        return list_of_unlabeled_turns
            
 
        
    def concatenate_texts(self, data_point):
        # Iterate through each item in data_point["dialog_formatted"] and concatenate
        dialog_formatted=concatenate_dialog(data_point["dialog_formatted"])

        
        all_extsums = []
        for ext_sum in data_point["extractive_summary"]:
            all_extsums.extend(ext_sum['sentences'])
        extractive_summary = ' '.join(all_extsums)
        
        all_absums = []
        for abs_sum in data_point["abstractive_summary"]:
            all_absums.extend(abs_sum['line'])
        abstractive_summary = ' '.join(all_absums)
        return dialog_formatted, extractive_summary, abstractive_summary
        
    def generate_and_tokenize_prompt2(self, data_point):
            #concatenate all sentences in for new format
            
            dialog_formatted, extractive_summary, abstractive_summary=self.concatenate_texts(data_point)

            full_prompt = (
                self.extractive_abstractive_prompt
                + " \n\n"
                + dialog_formatted
                + "\n\n"
                # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
                + "Extractive Summary and Abstractive Summary with References and Tags:"
                + "\n"
                + extractive_summary
                + "\n"
                + abstractive_summary
            )
            return self.tokenize(full_prompt)

    def generate_and_tokenize_prompt3(self, data_point):
        #concatenate all sentences in for new format
        
        # if type(data_point["dialog_formatted"]) is list:
        #     dialog_formatted, extractive_summary, abstractive_summary=self.concatenate_texts(data_point)
        # else:
        # dialog_formatted = data_point["dialog_formatted"]
        # extractive_summary = data_point["extractive_summaries_abhay"]
        # abstractive_summary = data_point["abstractive_summaries_generic_abhay"]
        
        full_prompt = (
            self.extractive_abstractive_prompt
            + " \n\n"
            + data_point['0']
            # + dialog_formatted
            # + "\n\n"
            # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
            # + "\n"
            # + extractive_summary
            # + "\n"
            # + abstractive_summary
        )
        return self.tokenize(full_prompt)
    # def concatenate_dialog(self, dialog):
    #     # if type(data_point["dialog_formatted"]) is list:
    #     all_sentences = []

    #     # Iterate through each item in data_point["dialog_formatted"]
    #     for item in dialog:
    #         # Extend the all_sentences list by adding sentences from the current item
    #         all_sentences.extend(item['sentences'])

    #     # Join all sentences into a single string, separated by a space (or any other separator you prefer)
    #     dialog_formatted = ' '.join(all_sentences)
    #     return dialog_formatted
    
    # def concatenate_dialog_w_newlines(self, dialog):
    #     # if type(data_point["dialog_formatted"]) is list:
    #     all_sentences = []

    #     # Iterate through each item in data_point["dialog_formatted"]
    #     for item in dialog:
    #         # Extend the all_sentences list by adding sentences from the current item
    #         all_sentences.extend(item['sentences'])
        
    #     # Join all sentences into a single string, separated by a space (or any other separator you prefer)
    #     dialog_formatted = '\n'.join(all_sentences)
    #     return dialog_formatted
    
    

    def invoke_extsum(self, dialog):
        eval_prompt = (
            self.extractive_prompt + " \n\n" + dialog + "\n\n" + "Extractive Summary:"
        )

        
        # model_input = self.model.tokenizer(
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
                self.model.generate(**model_input, max_new_tokens=200)[0],
                skip_special_tokens=True,
            )
            print(output)
            # extsum_output = output.split("Extractive Summary 2:")[1].strip()
            try:
                # extsum_output = output.split("Extractive Summary:")[1].strip()
                extsum_output = output.split("Extractive Summary:")[-1].strip()
            except:
                extsum_output = output.strip()

            return extsum_output

    def invoke_extsum_abssum(self, dialogue):
        eval_prompt = (
            self.extractive_abstractive_prompt
            + " \n\n"
            + dialogue
            + "\n\n"
            # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
            + self.prompt_ending
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
                self.model.generate(**model_input, max_new_tokens=400)[0],
                skip_special_tokens=True,
            )
            print(output)
            extsum_abssum_output = output.split(
                self.prompt_ending
            )[1].strip()

            return extsum_abssum_output

    def invoke_abssum_w_extsum(self, pred_extsum):
        eval_prompt = (
            self.abstractive_prompt
            + " \n\n"
            + pred_extsum
            + "\n\n"
            # + "Abstractive Summary 2 with References and Tags:"
            + "Abstractive Summary with References and Tags:"
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
                self.model.generate(**model_input, max_new_tokens=400)[0],
                skip_special_tokens=True,
            )
            print(output)
            try:
                abssum_output = output.split(
                    "Abstractive Summary with References and Tags:"
                )[1].strip()
            except:
                abssum_output = output.strip()

            return abssum_output
        
    def invoke_abssum_w_dialogue_and_extsum(self, dialogue, pred_extsum):
        eval_prompt = (
            self.abstractive_w_dialog_prompt
            + " \n\n"
            + dialogue
            + "\n\n"
            + "Extractive Summary:"
            + "\n\n"
            + pred_extsum
            + "\n\n"
            + "Abstractive Summary with References and Tags:"
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
                self.model.generate(**model_input, max_new_tokens=400)[0],
                skip_special_tokens=True,
            )
            print(output)
            try:
                abssum_output = output.split(
                    "Abstractive Summary with References and Tags:"
                )[-1].strip()
            except:
                abssum_output = output.strip()
            # if self.model_name == "citeXtral":
                # abssum_output = self.fix_formatting_finetuning_mismatch(abssum_output)
            return abssum_output
        
    def invoke_absum_w_oracle_extsum(self, oracle_extsum):
        eval_prompt = (
            self.abstractive_prompt
            + " \n\n"
            + oracle_extsum
            + "\n\n"
            # + "Abstractive Summary 2 with References and Tags:"
            + "Abstractive Summary with References and Tags:"
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
                self.model.generate(**model_input, max_new_tokens=400)[0],
                skip_special_tokens=True,
            )
            print(output)
            try:
                abssum_output = output.split(
                    "Abstractive Summary with References and Tags:"
                )[1].strip()
            except:
                abssum_output = output.strip()

            return abssum_output
        
    def fix_formatting_finetuning_mismatch(self, text):
        # Replace sequences of a letter followed by a dot and a space with a letter and a dot
        text_no_space_before_dot = text.replace('. ', '.  ')
        # Replace double spaces with a unique placeholder to preserve spaces between words
        text_with_placeholder = text_no_space_before_dot.replace('  ', '§')
        # Remove all remaining spaces (those that were separating characters)
        no_spaces = text_with_placeholder.replace(' ', '')
        # Replace the placeholder with a single space to restore spaces between words
        fixed_text = no_spaces.replace('§', ' ')
        return fixed_text
        
    def generate_and_tokenize_prompt2(self, data_point):
        #concatenate all sentences in for new format
        
        if type(data_point["dialog_formatted"]) is list:
            dialog_formatted, extractive_summary, abstractive_summary=self.concatenate_texts(data_point)
        else:
            dialog_formatted = data_point["dialog_formatted"]
            extractive_summary = data_point["extractive_summaries_abhay"]
            abstractive_summary = data_point["abstractive_summaries_generic_abhay"]
        
        full_prompt = (
            self.extractive_abstractive_prompt
            + " \n\n"
            + dialog_formatted
            + "\n\n"
            # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
            + "Extractive Summary and Abstractive Summary with References and Tags:"
            + "\n"
            + extractive_summary
            + "\n"
            + abstractive_summary
        )
        return self.tokenize(full_prompt)
    
    def generate_and_tokenize_training_prompt(self, data_point):
        dialog_formatted, extsum_string_with_numbers, absum_string_with_references_and_tags=self.format_text_for_training(data_point)
        full_prompt = (
            self.extractive_abstractive_prompt_no_few_shot
            + " \n\n"
            + dialog_formatted
            + "\n\n"
            # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
            + "Extractive Summary and Abstractive Summary with References and Tags:"
            + "\n"
            + extsum_string_with_numbers
            + "\n"
            + absum_string_with_references_and_tags
        )
        return self.tokenize(full_prompt)
    
    def generate_and_tokenize_training_prompt_for_unlabeled(self, data_point):
        full_prompt = (
            self.extractive_abstractive_prompt_no_few_shot
            + " \n\n"
            + data_point['dialogue']
            + "\n\n"
            # + "Extractive Summary 2 and Abstractive Summary 2 with References and Tags:"
            + "Extractive Summary and Abstractive Summary with References and Tags:"
            + "\n"
            + data_point['response']
        )
        return self.tokenize(full_prompt)
    
    def format_text_for_training(self, data_point):
        #concatenate all sentences in for new format
        dialog_formatted = concatenate_dialog(data_point['dialog_formatted'])
        
        extsum_list_with_numbers = []
        # for each extsum in the list of extractive summaries number them with n.- and concatenate with \n as separator
        for i, extsum in enumerate(data_point["extractive_summary"]):
            extsum_list_with_numbers.append(f"{i+1}.- {' '.join(extsum['sentences'])}")
        extsum_string_with_numbers = '\n'.join(extsum_list_with_numbers)

        absum_list_with_references_and_tags = []
        for abs_sum in data_point["abstractive_summary"]:
            # absum_list_with_references_and_tags.append("{"+f"\{abs_sum['line']}+"}")
            absum_list_with_references_and_tags.append(abs_sum['line'] +" {"+f"{','.join([str(line_id+1) for line_id in abs_sum['line_ids']])}"+"} "+"{"+f"{abs_sum['tag']}"+"}")
        absum_string_with_references_and_tags = '\n'.join(absum_list_with_references_and_tags)
        return dialog_formatted, extsum_string_with_numbers, absum_string_with_references_and_tags
    
        
