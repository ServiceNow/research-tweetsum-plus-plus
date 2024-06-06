from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer
import os
from factsumm.utils import handle_abstractive_predictions, concatenate_dialog
import pandas as pd


class GPTAbstractive():
    def __init__(self, model_name):
        self.tools = [TavilySearchResults(max_results=1)]
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.model_name = model_name
        self.llm = ChatOpenAI(model=self.model_name)
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)

        # Create an agent executor by passing in the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )
        
        with open(
            "prompts/only_abssum_prompt_1-shot_1.txt",
            "r",
        ) as file:
            abstractive_prompt = file.read()

        self.abstractive_prompt = abstractive_prompt
        
    def invoke_abssum(self, dialog):

        prompt = (
            self.abstractive_prompt
            + " \n\n"
            + concatenate_dialog(dialog)
            + "\n\n"
            + "Abstractive Summary:"
        )
        return self.agent_executor.invoke({"input": prompt})


    def predict(self):
        abssum_responses = []
        abssum_summaries = []
        for i, dialog in enumerate(self.test_dataset["dialog_formatted"]):
  
            abssum_response = self.invoke_abssum(dialog)
            abssum_responses.append(abssum_response)

            abssum_summaries.append(abssum_response["output"])
        self.predicted_abstractive_summaries = abssum_summaries
        if type(self.test_dataset) == pd.DataFrame:
                self.pandas_dataset = self.test_dataset
        else:
            self.pandas_dataset = self.test_dataset.to_pandas()

        self.pandas_dataset['raw_predicted_abstractive_summaries'] = self.predicted_abstractive_summaries
        self.pandas_dataset['handled_predicted_abstractive_summaries'] = handle_abstractive_predictions(self.pandas_dataset)
        
    def evaluate(self):
        if self.do_predict:
            self.predict()
        else:
            pass # un-enabling loading predictions for now
        (
            total_rouge1_precision_abstractive,
            total_rouge1_recall_abstractive,
            total_rouge1_fmeasure_abstractive,
        ) = (0, 0, 0)
        (
            total_rouge2_precision_abstractive,
            total_rouge2_recall_abstractive,
            total_rouge2_fmeasure_abstractive,
        ) = (0, 0, 0)
        (
            total_rougeL_precision_abstractive,
            total_rougeL_recall_abstractive,
            total_rougeL_fmeasure_abstractive,
        ) = (0, 0, 0)
        num_rows = 0
        # perform rouge evaluation of df
        for (
            index,
            row,
        ) in self.test_dataset.iterrows():
            num_rows += 1

            # Concatenate the summaries
            # Concatenate the summaries
            try:
                concatenated_reference_summary = " ".join(
                    [a["line"] for a in row["abstractive_summary"]]
                )
            except:
                concatenated_reference_summary = " ".join(row['abstractive_summary'])
                
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            abssum_scores_on_abs_gt = scorer.score(
                self.test_dataset.handled_predicted_abstractive_summaries[index],
                    concatenated_reference_summary,
                )
            # Accumulate ROUGE scores for abstractive summaries
            total_rouge1_precision_abstractive += abssum_scores_on_abs_gt["rouge1"][0]
            total_rouge1_recall_abstractive += abssum_scores_on_abs_gt["rouge1"][1]
            total_rouge1_fmeasure_abstractive += abssum_scores_on_abs_gt["rouge1"][2]

            total_rouge2_precision_abstractive += abssum_scores_on_abs_gt["rouge2"][0]
            total_rouge2_recall_abstractive += abssum_scores_on_abs_gt["rouge2"][1]
            total_rouge2_fmeasure_abstractive += abssum_scores_on_abs_gt["rouge2"][2]

            total_rougeL_precision_abstractive += abssum_scores_on_abs_gt["rougeL"][0]
            total_rougeL_recall_abstractive += abssum_scores_on_abs_gt["rougeL"][1]
            total_rougeL_fmeasure_abstractive += abssum_scores_on_abs_gt["rougeL"][2]
        
        avg_rouge1_abstractive = (
            total_rouge1_precision_abstractive / num_rows,
            total_rouge1_recall_abstractive / num_rows,
            total_rouge1_fmeasure_abstractive / num_rows,
        )
        avg_rouge2_abstractive = (
            total_rouge2_precision_abstractive / num_rows,
            total_rouge2_recall_abstractive / num_rows,
            total_rouge2_fmeasure_abstractive / num_rows,
        )
        avg_rougeL_abstractive = (
            total_rougeL_precision_abstractive / num_rows,
            total_rougeL_recall_abstractive / num_rows,
            total_rougeL_fmeasure_abstractive / num_rows,
        )
        
        print(
            f"Average ROUGE-1 scores for abstractive summaries: Precision={avg_rouge1_abstractive[0]}, Recall={avg_rouge1_abstractive[1]}, F-measure={avg_rouge1_abstractive[2]}"
        )
        print(
            f"Average ROUGE-2 scores for abstractive summaries: Precision={avg_rouge2_abstractive[0]}, Recall={avg_rouge2_abstractive[1]}, F-measure={avg_rouge2_abstractive[2]}"
        )
        print(
            f"Average ROUGE-L scores for abstractive summaries: Precision={avg_rougeL_abstractive[0]}, Recall={avg_rougeL_abstractive[1]}, F-measure={avg_rougeL_abstractive[2]}"
        )
        return (
                avg_rouge1_abstractive,
                avg_rouge2_abstractive,
                avg_rougeL_abstractive,
            )
        
    def train(self):
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
        self.project = "tweetsum-gpt-abstractive-api-call"
        self.base_model_name = "gpt_only_absum_prompt"
        # self.run_name = self.base_model_name + "-" + self.project
        self.run_name = self.save_dir + "-" + self.project + "-" + self.base_model_name
        # self.output_dir = "./" + self.run_name
        
        self.output_dir = (
            os.path.join(f"{self.base_path}/runs/", self.run_name)
        )
        os.makedirs(self.output_dir, exist_ok=True)
        