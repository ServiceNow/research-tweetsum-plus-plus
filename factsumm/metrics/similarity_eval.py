from evaluate import load
from factsumm.utils import get_references_from_line_numbers, handle_abs_pred_for_ref_sim

class SimilarityEval():
    def __init__(self) -> None:
        self.bertscore = load("bertscore")
    def evaluate(self, dataframe):
        num_rows = 0
        num_references = 0
        # perform rouge evaluation of df
        precision_list = []
        recall_list = []
        f1_list = []
        for (
            index,
            row,
        ) in dataframe.iterrows():
            num_rows += 1
            
            # # Concatenate the summaries
            # concatenated_reference_summary = " ".join(
            #     [a["Abs_sum"] for a in row["parsed_summaries"]]
            # )
            # references = get_references_from_line_numbers(dataframe.predicted_extractive_summaries[index], )
            references = get_references_from_line_numbers(dataframe.raw_predicted_extractive_summaries[index], dataframe.predicted_line_references[index])
            abstractive_predictions = handle_abs_pred_for_ref_sim(row)
            
            # if len(abstractive_predictions) != len(references):
            #     print("Length of abstractive predictions and references are not equal")
            #     print("Abstractive Predictions: ", abstractive_predictions)
            #     print("References: ", references)
            #     continue
            # bertscore_abs_vs_ref_dict = self.bertscore.compute(predictions=abstractive_predictions, references = references, lang='en')
            # num_references+=len(references)
            references_concatenated = " ".join(references)
            abstractive_predictions_concatenated = " ".join(abstractive_predictions)
            bertscore_abs_vs_ref_dict = self.bertscore.compute(predictions=[abstractive_predictions_concatenated], references = [references_concatenated], lang='en')
            precision_list.extend(bertscore_abs_vs_ref_dict['precision'])   
            recall_list.extend(bertscore_abs_vs_ref_dict['recall'])
            f1_list.extend(bertscore_abs_vs_ref_dict['f1'])
            print(bertscore_abs_vs_ref_dict)
        average_precision = sum(precision_list)/len(precision_list)
        average_recall = sum(recall_list)/len(recall_list)
        average_f1 = sum(f1_list)/len(f1_list)
        average_ref_per_row = num_references/num_rows
        print('average precision: ', average_precision)
        print('average recall: ', average_recall)
        print('average f1: ', average_f1)
        return average_precision, average_recall, average_f1, average_ref_per_row
            
            