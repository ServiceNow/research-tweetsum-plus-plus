from transformers import Trainer
from factsumm.utils import rouge_score

class RougeTrainer(Trainer):
    
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        loss = rouge_score(outputs, labels)
        return loss