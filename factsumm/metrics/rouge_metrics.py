import evaluate
rouge = evaluate.load("rouge")
# class RougeMetrics:
#     # def __init__(self, exp_dict=None, tokenizer=None):
#     #     self.exp_dict = exp_dict
#     #     self.tokenizer = tokenizer
#     def __init__(self):
#         self.metric = evaluate.load("rouge")
    
#     def compute_metrics(self, eval_pred):
#         # return {"rouge": self.rouge_score(pred)}
#         metric_key_prefix="validation"
#         eval_dict = {}
#         rouge_scores = self.metric.compute(predictions=gt_summaries, 
#                                            references=generated_summaries,
#                                            use_aggregator=True,
#                                            use_stemmer=True,
#                                            rouge_types = ["rouge1", "rouge2", "rougeL"]
#                                         ) 
#         eval_dict.update({f"{metric_key_prefix}_{k}": v.mid.fmeasure for k, v in rouge_scores.items()})
#         return eval_dict
         

def compute_metrics(eval_pred):
    # return {"rouge": self.rouge_score(pred)}
    predictions, labels = eval_pred
    # metric_key_prefix="validation"
    # eval_dict = {}
    # rouge_scores = self.metric.compute(predictions=gt_summaries, 
    #                                     references=generated_summaries,
    #                                     use_aggregator=True,
    #                                     use_stemmer=True,
    #                                     rouge_types = ["rouge1", "rouge2", "rougeL"]
    #                                 ) 
    # eval_dict.update({f"{metric_key_p
    # refix}_{k}": v.mid.fmeasure for k, v in rouge_scores.items()})
    
    
    
    return eval_dict
         