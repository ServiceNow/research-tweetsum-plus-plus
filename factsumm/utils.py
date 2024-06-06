import re
import hashlib
from rapidfuzz import process, fuzz
from datasets import load_metric


def clean_extractive_summaries(text):
    # remove the number and Agent: or Customer: from the extractive summaries
    if "Agent:" in text:
        text = text.split("Agent:")[1].strip()
    elif "Customer:" in text:
        text = text.split("Customer:")[1].strip()
    return text


def clean_abstractive_summaries(text):
    # remove the line references and tags from the abstractive summaries
    text = text.split("{")[0].strip()
    return text


def get_tag_from_response_line(text):
    # Regular expression to find tags within {}
    if "{" not in text or "}" not in text:
        return ""
    tag = text.split("{")[-1].split("}")[0].strip()
    return tag


def get_references_from_summaries(text):
    # Regular expression to find numbers within {}
    matches = re.findall(r"\{([\d,]+)\}", text)
    list_of_line_numbers = []
    # list_of_line_numbers = text.split("{")[1].split("}")[0].split(",")
    for match in matches:
        # Split numbers by comma and convert them to integers
        numbers = [int(num.strip()) for num in match.split(",")]
        # Extend the list with these numbers
        list_of_line_numbers.extend(numbers)
    return list_of_line_numbers


def handle_extractive_predictions(df):
    cleaned_conversations = []  # To store cleaned sentences of each conversation

    # Loop over each line in the Series object
    for lines in df["raw_predicted_extractive_summaries"]:
        current_clean_conversation = []
        for line in lines.split("\n"):
            cleaned_line = clean_extractive_summaries(
                line
            )  # Apply the cleaning function to the line
            current_clean_conversation.append(
                cleaned_line
            )  # Append the cleaned line to the list
        cleaned_conversations.append(" ".join(current_clean_conversation))

    return cleaned_conversations


def handle_abstractive_predictions(df):
    cleaned_conversations = []  # To store cleaned sentences of each conversation

    # Loop over each line in the Series object
    for lines in df["raw_predicted_abstractive_summaries"]:
        current_clean_conversation = []
        for line in lines.split("\n"):
            cleaned_line = clean_abstractive_summaries(
                line
            )  # Apply the cleaning function to the line
            current_clean_conversation.append(
                cleaned_line
            )  # Append the cleaned line to the list
        cleaned_conversations.append(" ".join(current_clean_conversation))

    return cleaned_conversations

def handle_abs_pred_for_ref_sim(row):
    clean_summaries = []
    # Loop over each line in the Series object
    for summary in row['raw_predicted_abstractive_summaries'].split('\n'):
        
        cleaned_line = clean_abstractive_summaries(
            summary
        )  # Apply the cleaning function to the line
        clean_summaries.append(
            cleaned_line
        )  # Append the cleaned line to the lis

    return clean_summaries


def handle_reference_predictions(df):
    list_of_list_of_references = []
    for lines in df["raw_predicted_abstractive_summaries"]:
        current_list_of_references = []
        for line in lines.split("\n"):
            list_of_references = get_references_from_summaries(line)
            # current_list_of_references.append(list_of_references)
            current_list_of_references.extend(list_of_references)
        list_of_list_of_references.append(current_list_of_references)
    return list_of_list_of_references


def create_extractive_summary_mapping(row):
    
    # Create a mapping of extractive summaries to the original conversations
    extsum_predictions_mapping = []
    extsum_gt_mapping = []

    predictions = row['raw_predicted_extractive_summaries']
    original_dialogues_list = row['dialog_formatted'].split("\n")

    extsum_predictions_list = predictions.strip(',').strip('\n').split('\n')
    # extractive_summaries_gt_list = row['extractive_summary']

    # create extsum predictions mapping to original dialogues
    for j, prediction in enumerate(extsum_predictions_list):
        best_match = process.extractOne(prediction, original_dialogues_list, scorer=fuzz.WRatio)
        extsum_predictions_mapping.append(best_match[-1])
    # extractive_summaries_gt = row['extractive_summaries_abhay'].split("\n")
    extractive_summaries_gt = concatenate_dialog(row['extractive_summary'])
    extractive_summaries_gt_list = extractive_summaries_gt.split('\n')
    # create extsum gt mapping to original dialogues
    for k, ext_sum_gt in enumerate(extractive_summaries_gt_list):
        best_match = process.extractOne(ext_sum_gt, original_dialogues_list, scorer=fuzz.WRatio)
        extsum_gt_mapping.append(best_match[-1])
    return extsum_predictions_mapping, extsum_gt_mapping

def handle_reference_predictions_fuzzy_match(df):
    list_of_list_of_pred_references = []
    
    for i, row in df.iterrows():
        extsum_predictions_mapping, _ = create_extractive_summary_mapping(row)
        cur_list_of_pred_references = []
        # get actual dialogue indices for the predicted abstractive summaries
        # if row["raw_predicted_abstractive_summaries"] is str:
        #     continue
        list_raw_predicted_abstractive_summaries = row["raw_predicted_abstractive_summaries"].split("\n")
        for line in list_raw_predicted_abstractive_summaries:
            list_of_references = get_references_from_summaries(line)
            # current_list_of_references.append(list_of_references)
            predicted_references_mapped = map_references_to_original_dialogue(list_of_references, extsum_predictions_mapping)
            # current_list_of_references.extend(list_of_references)
            cur_list_of_pred_references.extend(predicted_references_mapped)
        list_of_list_of_pred_references.append(cur_list_of_pred_references)
    return list_of_list_of_pred_references

def handle_reference_gt_fuzzy_match(df):
    list_of_list_of_gt_references = []
    
    for i, row in df.iterrows():
        _, extsum_gt_mapping = create_extractive_summary_mapping(row)
        cur_list_of_gt_references = []
        # get actual dialogue indices for the predicted abstractive summaries
        for line in row["abstractive_summary"]: #row["parsed_summaries"]:
            list_of_references = line["line_ids"] #line["Line Numbers"] #get_references_from_summaries(line)
            # current_list_of_references.append(list_of_references)
            gt_references_mapped = map_references_to_original_dialogue(list_of_references, extsum_gt_mapping)
            # current_list_of_references.extend(list_of_references)
            cur_list_of_gt_references.extend(gt_references_mapped)
        list_of_list_of_gt_references.append(cur_list_of_gt_references)
    return list_of_list_of_gt_references

def map_references_to_original_dialogue(list_of_references, mapping):
    # Given a list of references, map them to the extractive ground truth
    # This is a fuzzy match, so we will use the Levenshtein distance to find the closest match
    mapped_list = []
    for reference in list_of_references:
        if len(mapping)>=reference:
            mapped_list.append(mapping[reference-1]) # substracting one because the list of references is 1-indexed
    return mapped_list
        

def handle_tag_predictions(df):
    tags_per_conv = []  # To store cleaned sentences of each conversation

    # Loop over each line in the Series object
    for lines in df["raw_predicted_abstractive_summaries"]:
        current_tags = []
        for line in lines.split("\n"):
            tag = get_tag_from_response_line(
                line
            )  # Apply the cleaning function to the line
            if len(tag) == 0:
                continue
            current_tags.append(tag)  # Append the tag to the list
        # convs_with_tags.append(" ".join(current_tags))
        tags_per_conv.append(set(current_tags))

    return tags_per_conv

def hash_dict(exp_dict):
    """Create a hash for an experiment. Credtts to github.com/haven-ai!
 
    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters
 
    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")
 
    for k in sorted(exp_dict.keys()):
        if "." in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError(f"{exp_dict[k]} tuples can't be hashed yet, consider converting tuples to lists")
        elif isinstance(exp_dict[k], list) and len(exp_dict[k]) and isinstance(exp_dict[k][0], dict):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]
 
        dict2hash += str(k) + "/" + str(v)
    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()
 
    return hash_id

def get_references_from_line_numbers(extractive_summaries, line_numbers):
    # Given a line number, get the text of the extractive summary that starts with that number
    references = []
    for line_number in line_numbers:
        for summary in extractive_summaries.split("\n"): # review this
            if summary.startswith(str(line_number)):
                references.append(clean_extractive_summaries(summary))
                break
    return references

def concatenate_dialog(dialog):
    # if type(data_point["dialog_formatted"]) is list:
    all_sentences = []

    # Iterate through each item in data_point["dialog_formatted"]
    for item in dialog:
        # Extend the all_sentences list by adding sentences from the current item
        all_sentences.extend(item['sentences'])
    

    # Join all sentences into a single string, separated by a space (or any other separator you prefer)
    dialog_formatted = '\n'.join(all_sentences)
    return dialog_formatted


def rouge_score(predictions, targets):
    rouge_metric = load_metric("rouge")
    
    rouge_results = rouge_metric.compute(predictions=predictions, references=targets)
    return rouge_results["rouge2"].mid.fmeasure