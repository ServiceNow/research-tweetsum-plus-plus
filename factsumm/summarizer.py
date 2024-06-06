import pandas as pd
import json

from models.presumm_extractive import PreSummExtractive
from models.presumm_abstractive import PreSummAbstractive
from models.presumm import PreSumm
from models.mistralsumm import MistralSumm
from models.mixtralsumm import MixtralSumm
from models.citeXtral import CiteXtral
from factsumm.models.citeOPTsumm import CiteOptSumm
from factsumm.models.mixtralSelfLabel import MixtralSelfLabel
from models.optsumm import OPTSumm
from models.lead import Lead
from factsumm.utils import hash_dict
from models.gpt import GPT
from models.gpt_abstractive import GPTAbstractive
import yaml
from datasets import Dataset
import shutil
from factsumm.tweet_sum_processor import TweetSumProcessor


class Summarizer:

    def __init__(self, model_name, dataset_name, config_fname, load_weights=False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.load_weights = load_weights
        self.args = self.init_args(config_fname)
        self.train_dataset, self.test_dataset = (
            self.init_dataset()
        )  # add to huggingface datasets
        self.model = self.init_model()
        self.set_model_args()

    def init_args(self, config_fname):
        # Assuming your YAML file is named 'config.yaml'
        self.config_file_path = config_fname

        with open(self.config_file_path, "r") as file:
            args = yaml.safe_load(file)
        exp_id = hash_dict(args)
        self.save_dir = self.model_name + "_" + self.dataset_name + "_" + exp_id
        return args

    def init_model(self):
        if self.model_name == "presumm_extractive":
            return PreSummExtractive(self.args)
        elif self.model_name == "presumm_abstractive":
            return PreSummAbstractive(self.args)
        elif self.model_name == "presumm":
            return PreSumm(self.args)
        elif self.model_name == "mistralsumm":
            return MistralSumm()
        elif self.model_name == "mixtralsumm":
            return MixtralSumm()
        elif self.model_name == "lead":
            return Lead()
        elif self.model_name == "gpt-3.5-turbo-1106":
            return GPT(self.model_name)
        elif self.model_name == "gpt-4":
            return GPT(self.model_name)
        elif self.model_name == "optsumm":
            return OPTSumm()
        elif self.model_name == "gpt-3.5-turbo-1106/abstractive":
            return GPTAbstractive(self.model_name.split("/")[0])
        elif self.model_name == "gpt-4/abstractive":
            return GPTAbstractive(self.model_name.split("/")[0])
        elif self.model_name == "citeXtral":
            return CiteXtral()
        elif self.model_name == "citeOPT":
            return CiteOptSumm()
        elif self.model_name == "mixtral_self_label":
            return MixtralSelfLabel()

    def set_model_args(self):
        for key, value in self.args.items():
            setattr(self.model, key, value)
        self.model.train_dataset = self.train_dataset
        self.model.test_dataset = self.test_dataset
        self.model.save_dir = self.save_dir

    def summarize(self, text):
        return self.model.model.predict(text)

    def init_dataset(self):
        if self.dataset_name == "tweetsumm++":
            data = pd.read_excel("20240129_TweetSum_Abhay.xlsx")
            data = data.head(self.args["num_examples_to_evaluate"])
            # Convert the excel data to a DataFrame
            df = pd.DataFrame(data)

            df["parsed_summaries"] = df["abstractive_summaries_generic_abhay"].apply(
                self.parse_summary_improved
            )
            df = df[
                [
                    "dialog_formatted",
                    # "extractive_summaries1",
                    # "extractive_summaries2",
                    # "extractive_summaries3",
                    # "extractive_summaries4",
                    "extractive_summaries_abhay",
                    "abstractive_summaries_generic_abhay",
                    "parsed_summaries",
                ]
            ]
            if self.args["hf_dataset_format"]:
                # Convert the DataFrame to a HuggingFace Dataset
                return None, Dataset.from_pandas(df)
            else:
                return None, df
        elif self.dataset_name == "tweetsumm":
            processor = TweetSumProcessor("factsumm/Tweetsumm/twcs/twcs.csv")
            data_test = []
            data_training = []
            with open(
                "factsumm/Tweetsumm/tweet_sum_data_files/final_test_tweetsum.jsonl"
            ) as f:
                dialog_with_summaries = processor.get_dialog_with_summaries(
                    f.readlines()
                )
                # for i, dialog_with_summary in dialog_with_summaries:
                for i, dialog_with_summary in enumerate(dialog_with_summaries):
                    json_format = dialog_with_summary.get_json()
                    string_format = str(dialog_with_summary)
                    dialog_dict = json.loads(json_format)
                    data_test.append(dialog_dict)
                df_test = pd.DataFrame(data_test)
            with open(
                "factsumm/Tweetsumm/tweet_sum_data_files/final_train_tweetsum.jsonl"
            ) as f:
                train_dialog_with_summaries = processor.get_dialog_with_summaries(
                    f.readlines()
                )
                # for i, dialog_with_summary in dialog_with_summaries:
                for i, dialog_with_summary in enumerate(train_dialog_with_summaries):
                    try:
                        json_format = dialog_with_summary.get_json()
                        string_format = str(dialog_with_summary)
                        dialog_dict = json.loads(json_format)
                        data_training.append(dialog_dict)
                    except:
                        print(f"dialog {i} failed to parse")
                df_test = pd.DataFrame(data_test)
                df_training = pd.DataFrame(data_training)
                df_training["dialog_formatted"] = [
                    df_training["dialog"][i]["turns"]
                    for i in range(len(df_training["dialog"]))
                ]
                df_test["dialog_formatted"] = [
                    df_test["dialog"][i]["turns"] for i in range(len(df_test["dialog"]))
                ]
                # df_test["extractive_summary"] = [concatenate_dialog(df_test['summaries'][i]['extractive_summaries'][0]) for i in range(len(df_test['dialog'])) if len(df_test['summaries'][i]['extractive_summaries']) > 0 ]
                extractive_summaries = []
                abstractive_summaries = []
                for i in range(len(df_test["dialog"])):
                    try:
                        # extractive_summaries.append(concatenate_dialog(df_test['summaries'][i]['extractive_summaries'][0]))
                        extractive_summaries.append(
                            df_test["summaries"][i]["extractive_summaries"][0]
                        )
                        abstractive_summaries.append(
                            df_test["summaries"][i]["abstractive_summaries"][0]
                        )
                    except:
                        extractive_summaries.append("")
                        abstractive_summaries.append("")
                df_test["abstractive_summary"] = abstractive_summaries
                df_test["extractive_summary"] = extractive_summaries
                df_test = df_test.head(self.args["num_examples_to_evaluate"])
            if self.args["hf_dataset_format"]:
                return Dataset.from_pandas(df_training), Dataset.from_pandas(df_test)
            else:
                return df_training, df_test
        elif self.dataset_name == "tweetsumm++_v1":
            training_set_path = "tweetsumpp_v1_training_set.json"
            test_set_path = "tweetsumpp_v1_test_set.json"
            with open(training_set_path) as f1:
                training_set = json.load(f1)
            with open(test_set_path) as f2:
                test_set = json.load(f2)
            training_set = pd.DataFrame(training_set)
            training_set = training_set.head(self.args["num_examples_to_train"])
            test_set = pd.DataFrame(test_set)
            test_set = test_set.head(self.args["num_examples_to_evaluate"])
            if self.args["hf_dataset_format"]:
                return Dataset.from_pandas(training_set), Dataset.from_pandas(test_set)
            else:
                return training_set, test_set

        else:
            raise ValueError("Dataset not found")

    def parse_summary_improved(self, summary):
        # Check if the summary is a string
        if not isinstance(summary, str):
            return (
                []
            )  # Return an empty list or other default value for non-string inputs
        parts = summary.split("[SEP]")
        extracted_info = []
        for part in parts:
            part = part.strip()

            if "{" in part and "}" in part:
                # Extracting line numbers
                line_info_start = part.find("{") + 1
                line_info_end = part.find("}")
                line_info_str = part[line_info_start:line_info_end]
                line_numbers = [
                    int(num.strip().replace("Line ", ""))
                    for num in line_info_str.split(",")
                ]

                # Extracting tag and subtag
                tag_start = part.find("{", line_info_end) + 1
                tag_end = part.find("}", tag_start)
                tag_info = part[tag_start:tag_end].strip() if tag_start != 0 else ""

                subtag_start = part.find("{", tag_end) + 1
                subtag_end = part.find("}", subtag_start)
                subtag_info = (
                    part[subtag_start:subtag_end].strip() if subtag_start != 0 else ""
                )

                extracted_info.append(
                    {
                        "Abs_sum": part.split(".")[0],
                        "Line Numbers": line_numbers,
                        "Tag": tag_info,
                        "Subtag": subtag_info,
                    }
                )
                # add abstactive summary to the extracted info
        return extracted_info

    def run(self):
        self.model.train()
        shutil.copy(self.config_file_path, self.model.output_dir)
        print(self.model.output_dir)
        self.model.evaluate()


if __name__ == "__main__":

    # create arg parser that asks for model_name, dataset_name and config_file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--config_file", type=str, help="config file path")
    args = parser.parse_args()
    summarizer = Summarizer(args.model_name, args.dataset_name, args.config_file)

    summarizer.run()
    print("done!")
