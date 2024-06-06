from factsumm.models.backbone import Backbone
from factsumm.metrics.rouge_eval import RougeEval
import random
from factsumm.utils import (
    handle_extractive_predictions,
    handle_abstractive_predictions,
    handle_reference_predictions,
    handle_tag_predictions,
    handle_reference_predictions_fuzzy_match,
    handle_reference_gt_fuzzy_match,
    concatenate_dialog
)
import os
import pandas as pd
class Random(Backbone):
    def train(self):
        
        self.output_dir = "/mnt/colab_public/gebelangsn/factsumm/random"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    
    def predict(self):

        extsum_responses = []
        abssum_responses = []
        extsum_summaries = []
        abssum_summaries = []
        for i, dialog in enumerate(self.test_dataset["dialog_formatted"]):
            
            # select randomly two sentences from the dialog where is_agent is True and two sentences where is_agent is False
            extsum_sentences = random.sample(
                [' '.join(dialog[i]['sentences']) for i in range(len(dialog)) if dialog[i]['is_agent'] is True], 2) + random.sample(
                [' '.join(dialog[i]['sentences']) for i in range(len(dialog)) if dialog[i]['is_agent'] is False], 2
            )
            extsum_response = '\n'.join(extsum_sentences)
            abssum_response = extsum_response
            
            extsum_responses.append(extsum_response)
            extsum_summaries.append(extsum_response)

            abssum_responses.append(abssum_response)
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
        self.pandas_dataset['handled_predicted_extractive_summaries'] = handle_extractive_predictions(self.pandas_dataset)
        self.pandas_dataset['handled_predicted_abstractive_summaries'] = handle_abstractive_predictions(self.pandas_dataset)
            
    def evaluate(self):
        self.predict()
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

