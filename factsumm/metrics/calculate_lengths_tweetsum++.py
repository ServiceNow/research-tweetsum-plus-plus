import json
with open('/mnt/home/research-fact-summ/dialog_summaries_results.json') as file:
    data = json.load(file)

with open('/mnt/home/research-fact-summ/dialog_summaries_results_additional.json') as file:
    additional_data = json.load(file)
    data.extend(additional_data)

utterances_lengths_list = []
sentences_lenghts_list = []
tokens_lengths_list = []
utterances_lenghts_list_agent = []
sentences_lenghts_list_agent = []
tokens_lengths_list_agent = []
utterances_lenghts_list_user = []
sentences_lenghts_list_user = []
tokens_lengths_list_user = []


for dialog in data:
    # calculate average length of the dialogs in terms of the utterances (agent and user), sentences (agent and user) and tokens
    utterances_lengths_list.append(len(dialog['dialog_formatted']))
    for utterance in dialog['dialog_formatted']:
        sentences_lenghts_list.append(len(utterance['sentences']))
        utterances_lengths_list.append(len(utterance))
        if utterance['is_agent']:
            utterances_lenghts_list_agent.append(len(utterance))
            for sentence in utterance['sentences']:
                tokens_lengths_list_agent.append(len(sentence.split()))
                sentences_lenghts_list_user.append(len(sentence))
        else:
            utterances_lenghts_list_user.append(len(utterance))
            for sentence in utterance['sentences']:
                tokens_lengths_list_user.append(len(sentence.split()))
                sentences_lenghts_list_agent.append(len(sentence))
        
        
        for sentence in utterance['sentences']:
            tokens_lengths_list.append(len(sentence.split()))
            sentences_lenghts_list.append(len(sentence))
        
        utterances_lengths_list.append(len(utterance['sentences']))
    
    # for utterance in dialog['dialog_formatted']:
    #     sentences_lenghts_list.append(len(utterance['sentences']))
    #     utterances_lengths_list.append(len(utterance))
    #     if utterance['is_agent']:
    #         utterances_lenghts_list_agent.append(len(utterance))
    #         for sentence in utterance['sentences']:
    #             tokens_lengths_list_agent.append(len(sentence.split()))
    #             sentences_lenghts_list_user.append(len(sentence))
    #     else:
    #         utterances_lenghts_list_user.append(len(utterance))
    #         for sentence in utterance['sentences']:
    #             tokens_lengths_list_user.append(len(sentence.split()))
    #             sentences_lenghts_list_agent.append(len(sentence))
        
        
    #     for sentence in utterance['sentences']:
    #         tokens_lengths_list.append(len(sentence.split()))
    #         sentences_lenghts_list.append(len(sentence))
        
    #     utterances_lengths_list.append(len(utterance['sentences']))

#print average lengths and standard deviations
print('Average utterances length:', sum(utterances_lengths_list) / len(utterances_lengths_list))
print('Average sentences length:', sum(sentences_lenghts_list) / len(sentences_lenghts_list))
print('Average tokens length:', sum(tokens_lengths_list) / len(tokens_lengths_list))
print('Average utterances length agent:', sum(utterances_lenghts_list_agent) / len(utterances_lenghts_list_agent))
print('Average sentences length agent:', sum(sentences_lenghts_list_agent) / len(sentences_lenghts_list_agent))
print('Average tokens length agent:', sum(tokens_lengths_list_agent) / len(tokens_lengths_list_agent))
print('Average utterances length user:', sum(utterances_lenghts_list_user) / len(utterances_lenghts_list_user))
print('Average sentences length user:', sum(sentences_lenghts_list_user) / len(sentences_lenghts_list_user))
print('Average tokens length user:', sum(tokens_lengths_list_user) / len(tokens_lengths_list_user))

import numpy as np
print('Standard deviation utterances length:', np.std(utterances_lengths_list))
print('Standard deviation sentences length:', np.std(sentences_lenghts_list))
print('Standard deviation tokens length:', np.std(tokens_lengths_list))
print('Standard deviation utterances length agent:', np.std(utterances_lenghts_list_agent))
print('Standard deviation sentences length agent:', np.std(sentences_lenghts_list_agent))
print('Standard deviation tokens length agent:', np.std(tokens_lengths_list_agent))
print('Standard deviation utterances length user:', np.std(utterances_lenghts_list_user))
print('Standard deviation sentences length user:', np.std(sentences_lenghts_list_user))
print('Standard deviation tokens length user:', np.std(tokens_lengths_list_user))
