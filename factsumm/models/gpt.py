from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from datasets import load_dataset, Dataset
from datetime import datetime
import pandas as pd
from factsumm.metrics.rouge_eval import RougeEval
from factsumm.metrics.prf1_eval import PRF1
from factsumm.metrics.similarity_eval import SimilarityEval
from factsumm.utils import (
    handle_extractive_predictions,
    handle_abstractive_predictions,
    handle_reference_predictions,
    handle_tag_predictions,
    concatenate_dialog
)
import os
from factsumm.models.backbone import Backbone

class GPT(Backbone):
    def __init__(self, model_name):
        self.tools = [TavilySearchResults(max_results=1)]
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.model_name = model_name
        self.llm = ChatOpenAI(model=self.model_name)
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )

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
            "prompts/abssum_w_dialogue_prompt_2-shot_1.txt",
            "r"
        ) as file:
            abstractive_w_dialog_prompt = file.read()
            
        self.abstractive_w_dialog_prompt = abstractive_w_dialog_prompt


    def invoke(self, text):
        return self.agent_executor.invoke({"input": text})

    def invoke_extsum(self, current_dialogue):
        
        if type(current_dialogue) is list:
            current_dialogue = concatenate_dialog(current_dialogue)
        prompt = (
            self.extractive_prompt
            + " \n\n"
            + current_dialogue
            + "\n\n"
            # + "Extractive Summary 2:"
            "Extractive Summary:"
        )

        return self.agent_executor.invoke({"input": prompt})
    
    def invoke_abssum_w_dialogue_and_extsum(self, dialogue, pred_extsum):
        prompt = (
            self.abstractive_w_dialog_prompt
            + " \n\n"
            + concatenate_dialog(dialogue)
            + "\n\n"
            + "Extractive Summary:"
            + "\n\n"
            + pred_extsum
            + "\n\n"
            + "Abstractive Summary with References and Tags:"
        )
        return self.agent_executor.invoke({"input": prompt})

    def invoke_abssum_w_pred_extsum(self, pred_extsum):

        prompt = (
            self.abstractive_prompt
            + " \n\n"
            # + pred_extsum["output"]
            + pred_extsum
            + "\n\n"
            + "Abstractive Summary with References and Tags:"
        )
        return self.agent_executor.invoke({"input": prompt})
    def clean_extsum_string(self, text):
        # search for \n\n1 in string
        import re
        id = re.search(r"\n\n\d", text)
        id2 = re.search(r"\n\n-End of Response-", text)
        # remove all before id
        if id:
            text = text[id.start() + 2 :]
        if id2:
            text = text[:id2.start()]
        start_of_extsum = '\n1.-'
        if start_of_extsum in text:
            text.split(start_of_extsum)[1]
        text = text.replace("Extractive Summary:\n", "")
        text = text.replace("\n\n-End of Response-", "")
        return text
    
        
    def predict(self):
        extsum_responses = []
        abssum_responses = []
        extsum_summaries = []
        abssum_summaries = []
        for i, dialog in enumerate(self.test_dataset["dialog_formatted"]):
            extsum_response = self.invoke_extsum(dialog)
            extsum_responses.append(extsum_response)
            clean_extsum = self.clean_extsum_string(extsum_response["output"])
            extsum_summaries.append(clean_extsum)
            
            # abssum_response = self.invoke_abssum_w_pred_extsum(clean_extsum)
            abssum_response = self.invoke_abssum_w_dialogue_and_extsum(dialog, clean_extsum)
            abssum_responses.append(abssum_response)

            if (
                "Tags:" in abssum_response["output"]
            ):
                abssum_summaries.append(
                    abssum_response["output"].split(
                        "Tags:"
                    )[1].strip().strip("-End of Response-").strip()
                )
            else:
                abssum_summaries.append(abssum_response["output"].strip("-End of Response-").strip())
        self.predicted_extractive_summaries = extsum_summaries
        self.predicted_abstractive_summaries = abssum_summaries
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # # convert list into a new pandas dataframe
        # df_summaries = pd.DataFrame(
        #     {
        #         "predicted_extractive_summaries": self.predicted_extractive_summaries,
        #         "predicted_abstractive_summaries": self.predicted_abstractive_summaries,
        #     }
        # )
        if type(self.test_dataset) == pd.DataFrame:
            self.pandas_dataset = self.test_dataset
        else:
            self.pandas_dataset = self.test_dataset.to_pandas()
        # df_summaries = pd.DataFrame(
        #     {
        #         "predicted_extractive_summaries": self.predicted_extractive_summaries,
        #         "predicted_abstractive_summaries": self.predicted_abstractive_summaries,
        #     }
        # )
        #TODO: first parse the references and then clean the extractive summaries from the line numbers
        self.pandas_dataset['raw_predicted_extractive_summaries'] = self.predicted_extractive_summaries
        self.pandas_dataset['raw_predicted_abstractive_summaries'] = self.predicted_abstractive_summaries
        self.pandas_dataset['predicted_line_references'] = handle_reference_predictions(self.pandas_dataset)
        self.pandas_dataset['predicted_tags'] = handle_tag_predictions(self.pandas_dataset)
        self.pandas_dataset['handled_predicted_extractive_summaries'] = handle_extractive_predictions(self.pandas_dataset)
        self.pandas_dataset['handled_predicted_abstractive_summaries'] = handle_abstractive_predictions(self.pandas_dataset)
        # df_summaries['raw_predicted_extractive_summaries'] = self.predicted_extractive_summaries #
        # df_summaries['predicted_extractive_summaries'] = handle_extractive_predictions(df_summaries) #
        # df_summaries['raw_predicted_abstractive_summaries'] = self.predicted_abstractive_summaries
        # df_summaries['handled_predicted_abstractive_summaries'] = handle_abstractive_predictions(df_summaries)
        # self.pred_df = df_summaries

        # # save dataframe
        # df_summaries.to_csv(
        #     f"{self.output_dir}/predicted_summaries_{self.model_name}_{timestamp}.csv",
        #     index=False,
        # )
        self.pandas_dataset.to_csv(
            # f"/mnt/colab_public/data/gebelangsn/factsumm/outputs/predicted_summaries_{self.model_name}_{timestamp}.csv",
            os.path.join(self.output_dir, "predicted_summaries.csv"),
            index=False,
        )
        
    def train(self):
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
        self.project = "tweetsum-gpt-api-call"
        self.base_model_name = "gpt_extsum_prompt_absum_prompt"
        # self.run_name = self.base_model_name + "-" + self.project
        self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name
        # self.output_dir = "./" + self.run_name
        
        self.output_dir = (
            os.path.join(f"{self.base_path}/runs/", self.run_name)
        )
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except PermissionError:
            self.output_dir = (
            os.path.join(f"{self.base_data_path_remote}/runs/", self.run_name)
        )
            