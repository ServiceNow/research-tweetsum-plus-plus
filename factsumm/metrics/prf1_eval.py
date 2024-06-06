import numpy as np
from collections import defaultdict


class PRF1:
    def __init__(self):
        pass

    def calculate_metrics(self, ground_truth, predictions):
        # Convert lists to sets for efficient searching
        ground_truth_set = set(ground_truth)
        prediction_set = set(predictions)

        # Calculate TP, FP, FN
        tp = len(prediction_set.intersection(ground_truth_set))
        fp = len(prediction_set - ground_truth_set)
        fn = len(ground_truth_set - prediction_set)

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0
        )

        return precision, recall, f1

    def evaluate(self, df):
        num_rows = 0

        precision_list = []
        recall_list = []
        f1_list = []

        # perform rouge evaluation of df
        for (
            index,
            row,
        ) in (
            df.iterrows()
        ):  # modify this to evaluate the entire dataset when test set is ready
            num_rows += 1
            # concatenated_list = []
            # for sum in row["parsed_summaries"]:
            #     concatenated_list.extend(sum["Line Numbers"])
            

            precision_parallel, recall_parallel, f1_parallel = self.calculate_metrics(
                # concatenated_list, df["predicted_line_references"][index]
                df["ground_truth_line_references"][index], df["predicted_line_references"][index]
            )

            precision_list.append(precision_parallel)
            recall_list.append(recall_parallel)
            f1_list.append(f1_parallel)

        print(
            "Average random Precision: ",
            np.array(precision_list).sum() / num_rows,
        )
        print("Average random Recall: ", np.array(recall_list).sum() / num_rows)
        print("Average random F1: ", np.array(f1_list).sum() / num_rows)

        print("done!")
        average_random_precision = np.array(precision_list).sum() / num_rows
        average_random_recall = np.array(recall_list).sum() / num_rows
        average_random_f1 = np.array(f1_list).sum() / num_rows
        return average_random_precision, average_random_recall, average_random_f1

    def evaluate_per_tag(self, df):
        num_rows = 0

        precision_list_RESOLUTION = []
        precision_list_RESOLUTION_STEP = []
        precision_list_WORKAROUND = []
        precision_list_ISSUE = []
        precision_list_DEFLECTION = []
        precision_list_FEEDBACK = []
        precision_list_QUESTION = []

        recall_list_RESOLUTION = []
        recall_list_RESOLUTION_STEP = []
        recall_list_WORKAROUND = []
        recall_list_ISSUE = []
        recall_list_DEFLECTION = []
        recall_list_FEEDBACK = []
        recall_list_QUESTION = []

        f1_list_RESOLUTION = []
        f1_list_RESOLUTION_STEP = []
        f1_list_WORKAROUND = []
        f1_list_ISSUE = []
        f1_list_DEFLECTION = []
        f1_list_FEEDBACK = []
        f1_list_QUESTION = []

        num_rows_w_RESOLUTION = 0
        num_rows_w_RESOLUTION_STEP = 0
        num_rows_w_WORKAROUND = 0
        num_rows_w_ISSUE = 0
        num_rows_w_DEFLECTION = 0
        num_rows_w_FEEDBACK = 0
        num_rows_w_QUESTION = 0

        # perform rouge evaluation of df
        for (
            index,
            row,
        ) in (
            df.iterrows()
        ):  # modify this to evaluate the entire dataset when test set is ready
            num_rows += 1
            concatenated_list = []
            for sum in row['abstractive_summary']: #row["parsed_summaries"]:
                concatenated_list.append(sum["tag"].upper())

            metrics = self.calculate_metrics_per_tag(
                concatenated_list, df["predicted_tags"][index]
            )

            for tag, values in metrics.items():
                if tag == "":
                    continue
                elif tag == "None":
                    continue
                elif tag == "RESOLUTION":
                    precision_list_RESOLUTION.append(values["precision"])
                    recall_list_RESOLUTION.append(values["recall"])
                    f1_list_RESOLUTION.append(values["f1"])
                    num_rows_w_RESOLUTION += 1
                elif tag == "RESOLUTION STEP":
                    precision_list_RESOLUTION_STEP.append(values["precision"])
                    recall_list_RESOLUTION_STEP.append(values["recall"])
                    f1_list_RESOLUTION_STEP.append(values["f1"])
                    num_rows_w_RESOLUTION_STEP += 1
                elif tag == "WORKAROUND":
                    precision_list_WORKAROUND.append(values["precision"])
                    recall_list_WORKAROUND.append(values["recall"])
                    f1_list_WORKAROUND.append(values["f1"])
                    num_rows_w_WORKAROUND += 1
                elif tag == "ISSUE":
                    precision_list_ISSUE.append(values["precision"])
                    recall_list_ISSUE.append(values["recall"])
                    f1_list_ISSUE.append(values["f1"])
                    num_rows_w_ISSUE += 1
                elif tag == "DEFLECTION":
                    precision_list_DEFLECTION.append(values["precision"])
                    recall_list_DEFLECTION.append(values["recall"])
                    f1_list_DEFLECTION.append(values["f1"])
                    num_rows_w_DEFLECTION += 1
                elif tag == "FEEDBACK":
                    precision_list_FEEDBACK.append(values["precision"])
                    recall_list_FEEDBACK.append(values["recall"])
                    f1_list_FEEDBACK.append(values["f1"])
                    num_rows_w_FEEDBACK += 1
                elif tag == "QUESTION":
                    precision_list_QUESTION.append(values["precision"])
                    recall_list_QUESTION.append(values["recall"])
                    f1_list_QUESTION.append(values["f1"])
                    num_rows_w_QUESTION += 1
        print(
            "Average TAG F1_RESOLUTION: ",
            np.array(f1_list_RESOLUTION).sum() / num_rows_w_RESOLUTION,
        )
        print(
            "Average TAG F1_RESOLUTION_STEP: ",
            np.array(f1_list_RESOLUTION_STEP).sum() / num_rows_w_RESOLUTION_STEP,
        )

        print(
            "Average TAG F1_WORKAROUND: ",
            np.array(f1_list_WORKAROUND).sum() / num_rows_w_WORKAROUND,
        )
        print(
            "Average TAG F1_ISSUE: ", np.array(f1_list_ISSUE).sum() / num_rows_w_ISSUE
        )

        print(
            "Average TAG F1_DEFLECTION: ",
            np.array(f1_list_DEFLECTION).sum() / num_rows_w_DEFLECTION,
        )

        print(
            "Average TAG F1_FEEDBACK: ",
            np.array(f1_list_FEEDBACK).sum() / num_rows_w_FEEDBACK,
        )
        print(
            "Average TAG F1_QUESTION: ",
            np.array(f1_list_QUESTION).sum() / num_rows_w_QUESTION,
        )
        

        print("done!")
        average_tag_f1_RESOLUTION = (
            np.array(f1_list_RESOLUTION).sum() / num_rows_w_RESOLUTION
        )
        average_tag_f1_RESOLUTION_STEP = (
            np.array(f1_list_RESOLUTION_STEP).sum() / num_rows_w_RESOLUTION_STEP
        )
        average_tag_f1_WORKAROUND = (
            np.array(f1_list_WORKAROUND).sum() / num_rows_w_WORKAROUND
        )
        average_tag_f1_ISSUE = np.array(f1_list_ISSUE).sum() / num_rows_w_ISSUE
        average_tag_f1_DEFLECTION = (
            np.array(f1_list_DEFLECTION).sum() / num_rows_w_DEFLECTION
        )
        average_tag_f1_FEEDBACK = np.array(f1_list_FEEDBACK).sum() / num_rows_w_FEEDBACK
        average_tag_f1_QUESTION = np.array(f1_list_QUESTION).sum() / num_rows_w_QUESTION
        #calculate average f1 of all tags
        average_f1_overall = (average_tag_f1_RESOLUTION+average_tag_f1_RESOLUTION_STEP+average_tag_f1_WORKAROUND+average_tag_f1_ISSUE+average_tag_f1_DEFLECTION+average_tag_f1_FEEDBACK+average_tag_f1_QUESTION)/7
        return (
            average_tag_f1_RESOLUTION,
            average_tag_f1_RESOLUTION_STEP,
            average_tag_f1_WORKAROUND,
            average_tag_f1_ISSUE,
            average_tag_f1_DEFLECTION,
            average_tag_f1_FEEDBACK,
            average_tag_f1_QUESTION,
            average_f1_overall
        )

    def calculate_metrics_per_tag(self, ground_truths, predictions):
        # Initializing dictionaries to hold count of true positives, false positives, and false negatives
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        # Iterate over each set of tags to update counts
        # for true_tags, predicted_tags in zip(ground_truths, predictions):
        # true_tags = ground_truths
        # predicted_tags = predictions
        true_set, predicted_set = set(ground_truths), set(predictions)

        # Update true positives and false negatives
        for tag in true_set:
            if tag in predicted_set:
                tp[tag] += 1
            else:
                fn[tag] += 1

        # Update false positives
        for tag in predicted_set:
            if tag not in true_set:
                fp[tag] += 1

        # Calculate precision, recall, and F1 score for each tag
        metrics = {}
        for tag in set(tp) | set(fp) | set(fn):
            precision = tp[tag] / (tp[tag] + fp[tag]) if tp[tag] + fp[tag] > 0 else 0
            recall = tp[tag] / (tp[tag] + fn[tag]) if tp[tag] + fn[tag] > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )

            metrics[tag] = {"precision": precision, "recall": recall, "f1": f1}

        return metrics
