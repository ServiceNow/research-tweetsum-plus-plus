from rouge_score import rouge_scorer
from datetime import datetime
from factsumm.utils import clean_extractive_summaries, clean_abstractive_summaries


class RougeEvalExtractive:
    def __init__(self) -> None:
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate(self, dataframe):
        # initialize variable to accumulate scores of extractive summaries
        (
            total_rouge1_precision_extractive,
            total_rouge1_recall_extractive,
            total_rouge1_fmeasure_extractive,
        ) = (0, 0, 0)
        (
            total_rouge2_precision_extractive,
            total_rouge2_recall_extractive,
            total_rouge2_fmeasure_extractive,
        ) = (0, 0, 0)
        (
            total_rougeL_precision_extractive,
            total_rougeL_recall_extractive,
            total_rougeL_fmeasure_extractive,
        ) = (0, 0, 0)

        # initialize variable to accumulate scores of abstractive summaries

        (
            total_rouge1_precision_extsum_on_ext_gt,
            total_rouge2_precision_extsum_on_ext_gt,
            total_rougeL_precision_extsum_on_ext_gt,
        ) = (0, 0, 0)

        (
            total_rouge1_recall_extsum_on_ext_gt,
            total_rouge2_recall_extsum_on_ext_gt,
            total_rougeL_recall_extsum_on_ext_gt,
        ) = (0, 0, 0)

        (
            total_rouge1_fmeasure_extsum_on_ext_gt,
            total_rouge2_fmeasure_extsum_on_ext_gt,
            total_rougeL_fmeasure_extsum_on_ext_gt,
        ) = (0, 0, 0)

        num_rows = 0
        # perform rouge evaluation of df
        for (
            index,
            row,
        ) in dataframe.iterrows():
            num_rows += 1

            # Concatenate the summaries
            concatenated_reference_summary = " ".join(
                [a["Abs_sum"] for a in row["parsed_summaries"]]
            )
            concatenated_extsum_summary = " ".join(
                [
                    clean_extractive_summaries(a)
                    for a in row["extractive_summaries_abhay"].split("[SEP]")
                ]
            )

            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            extsum_scores_on_ext_gt = scorer.score(
                dataframe.handled_predicted_extractive_summaries[index],
                concatenated_extsum_summary,
            )

            exstum_scores_on_abs_gt = scorer.score(
                dataframe.handled_predicted_extractive_summaries[index],
                concatenated_reference_summary,
            )
            # abssum_scores_on_abs_gt = scorer.score(
            #     dataframe.handled_predicted_abstractive_summaries[index],
            #     concatenated_reference_summary,
            # )

            # You can now use 'scores' which is a dictionary containing ROUGE scores
            print(f"Scores for row {index}: {exstum_scores_on_abs_gt}")
            # Accumulate ROUGE scores for extractive summaries
            total_rouge1_precision_extractive += exstum_scores_on_abs_gt["rouge1"][0]
            total_rouge1_recall_extractive += exstum_scores_on_abs_gt["rouge1"][1]
            total_rouge1_fmeasure_extractive += exstum_scores_on_abs_gt["rouge1"][2]

            total_rouge2_precision_extractive += exstum_scores_on_abs_gt["rouge2"][0]
            total_rouge2_recall_extractive += exstum_scores_on_abs_gt["rouge2"][1]
            total_rouge2_fmeasure_extractive += exstum_scores_on_abs_gt["rouge2"][2]

            total_rougeL_precision_extractive += exstum_scores_on_abs_gt["rougeL"][0]
            total_rougeL_recall_extractive += exstum_scores_on_abs_gt["rougeL"][1]
            total_rougeL_fmeasure_extractive += exstum_scores_on_abs_gt["rougeL"][2]

            # Accumulate ROUGE scores for abstractive summaries
            # total_rouge1_precision_abstractive += abssum_scores_on_abs_gt["rouge1"][0]
            # total_rouge1_recall_abstractive += abssum_scores_on_abs_gt["rouge1"][1]
            # total_rouge1_fmeasure_abstractive += abssum_scores_on_abs_gt["rouge1"][2]

            # total_rouge2_precision_abstractive += abssum_scores_on_abs_gt["rouge2"][0]
            # total_rouge2_recall_abstractive += abssum_scores_on_abs_gt["rouge2"][1]
            # total_rouge2_fmeasure_abstractive += abssum_scores_on_abs_gt["rouge2"][2]

            # total_rougeL_precision_abstractive += abssum_scores_on_abs_gt["rougeL"][0]
            # total_rougeL_recall_abstractive += abssum_scores_on_abs_gt["rougeL"][1]
            # total_rougeL_fmeasure_abstractive += abssum_scores_on_abs_gt["rougeL"][2]

            # Accumulate ROUGE scores for extractive summaries
            total_rouge1_precision_extsum_on_ext_gt += extsum_scores_on_ext_gt[
                "rouge1"
            ][0]
            total_rouge2_precision_extsum_on_ext_gt += extsum_scores_on_ext_gt[
                "rouge2"
            ][0]
            total_rougeL_precision_extsum_on_ext_gt += extsum_scores_on_ext_gt[
                "rougeL"
            ][0]

            total_rouge1_recall_extsum_on_ext_gt += extsum_scores_on_ext_gt["rouge1"][1]
            total_rouge2_recall_extsum_on_ext_gt += extsum_scores_on_ext_gt["rouge2"][1]
            total_rougeL_recall_extsum_on_ext_gt += extsum_scores_on_ext_gt["rougeL"][1]

            total_rouge1_fmeasure_extsum_on_ext_gt += extsum_scores_on_ext_gt["rouge1"][
                2
            ]
            total_rouge2_fmeasure_extsum_on_ext_gt += extsum_scores_on_ext_gt["rouge2"][
                2
            ]
            total_rougeL_fmeasure_extsum_on_ext_gt += extsum_scores_on_ext_gt["rougeL"][
                2
            ]

        # Calculate average scores
        avg_rouge1_extractive = (
            total_rouge1_precision_extractive / num_rows,
            total_rouge1_recall_extractive / num_rows,
            total_rouge1_fmeasure_extractive / num_rows,
        )
        avg_rouge2_extractive = (
            total_rouge2_precision_extractive / num_rows,
            total_rouge2_recall_extractive / num_rows,
            total_rouge2_fmeasure_extractive / num_rows,
        )
        avg_rougeL_extractive = (
            total_rougeL_precision_extractive / num_rows,
            total_rougeL_recall_extractive / num_rows,
            total_rougeL_fmeasure_extractive / num_rows,
        )

        # avg_rouge1_abstractive = (
        #     total_rouge1_precision_abstractive / num_rows,
        #     total_rouge1_recall_abstractive / num_rows,
        #     total_rouge1_fmeasure_abstractive / num_rows,
        # )
        # avg_rouge2_abstractive = (
        #     total_rouge2_precision_abstractive / num_rows,
        #     total_rouge2_recall_abstractive / num_rows,
        #     total_rouge2_fmeasure_abstractive / num_rows,
        # )
        # avg_rougeL_abstractive = (
        #     total_rougeL_precision_abstractive / num_rows,
        #     total_rougeL_recall_abstractive / num_rows,
        #     total_rougeL_fmeasure_abstractive / num_rows,
        # )

        avg_rouge1_extsum_on_ext_gt = (
            total_rouge1_precision_extsum_on_ext_gt / num_rows,
            total_rouge1_recall_extsum_on_ext_gt / num_rows,
            total_rouge1_fmeasure_extsum_on_ext_gt / num_rows,
        )
        avg_rouge2_extsum_on_ext_gt = (
            total_rouge2_precision_extsum_on_ext_gt / num_rows,
            total_rouge2_recall_extsum_on_ext_gt / num_rows,
            total_rouge2_fmeasure_extsum_on_ext_gt / num_rows,
        )
        avg_rougeL_extsum_on_ext_gt = (
            total_rougeL_precision_extsum_on_ext_gt / num_rows,
            total_rougeL_recall_extsum_on_ext_gt / num_rows,
            total_rougeL_fmeasure_extsum_on_ext_gt / num_rows,
        )

        # print(
        #     f"Average ROUGE-1 scores for abstractive summaries: Precision={avg_rouge1_abstractive[0]}, Recall={avg_rouge1_abstractive[1]}, F-measure={avg_rouge1_abstractive[2]}"
        # )
        # print(
        #     f"Average ROUGE-2 scores for abstractive summaries: Precision={avg_rouge2_abstractive[0]}, Recall={avg_rouge2_abstractive[1]}, F-measure={avg_rouge2_abstractive[2]}"
        # )
        # print(
        #     f"Average ROUGE-L scores for abstractive summaries: Precision={avg_rougeL_abstractive[0]}, Recall={avg_rougeL_abstractive[1]}, F-measure={avg_rougeL_abstractive[2]}"
        # )
        print(
            f"Average ROUGE-1 scores for extractive summaries on extractive ground truth: Precision={avg_rouge1_extsum_on_ext_gt[0]}, Recall={avg_rouge1_extsum_on_ext_gt[1]}, F-measure={avg_rouge1_extsum_on_ext_gt[2]}"
        )
        print(
            f"Average ROUGE-2 scores for extractive summaries on extractive ground truth: Precision={avg_rouge2_extsum_on_ext_gt[0]}, Recall={avg_rouge2_extsum_on_ext_gt[1]}, F-measure={avg_rouge2_extsum_on_ext_gt[2]}"
        )
        print(
            f"Average ROUGE-L scores for extractive summaries on extractive ground truth: Precision={avg_rougeL_extsum_on_ext_gt[0]}, Recall={avg_rougeL_extsum_on_ext_gt[1]}, F-measure={avg_rougeL_extsum_on_ext_gt[2]}"
        )

        return (
            avg_rouge1_extractive,
            avg_rouge2_extractive,
            avg_rougeL_extractive,
            avg_rouge1_extsum_on_ext_gt,
            avg_rouge2_extsum_on_ext_gt,
            avg_rougeL_extsum_on_ext_gt,
        )
