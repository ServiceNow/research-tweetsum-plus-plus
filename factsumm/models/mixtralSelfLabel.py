import transformers
from datetime import datetime
from factsumm.models.mixtralsumm import MixtralSumm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from factsumm.models.mixtralsumm import MixtralSumm

class MixtralSelfLabel(MixtralSumm):
    def __init__(self):
        self.model_name = "mixtral_selflabel"
        self.max_length = 10000 #2400  # verify if need to increase based on dataset
        self.base_model_id = "mistralai/Mixtral-8x7B-v0.1"  # "facebook/opt-125m"  # "mistralai/Mixtral-8x7B-v0.1"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, quantization_config=self.bnb_config, device_map="auto"
        )
        # self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, add_eos_token=True, add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, add_bos_token=True, trust_remote_code=True
        )
        self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
        self.model_name = "mixtral_self_label"
        
        self.model.load_adapter(self.checkpoint_path)
        
    def train(self, args):
        # train model
        self.init_unlabeled_dataset()
        if self.local_run:
            self.base_path = self.base_data_path_local
        else:
            self.base_path = self.base_data_path_remote
            
        # for unlabeled_turns in self.list_of_unlabeled_turns:
        #     self_labeled_dialog=self.invoke_extsum_abssum(unlabeled_turns)
        #     self_labeled_confidence = self.invoke_assessment(self_labeled_dialog)
        #     self_labeled_dialogs.append(self_labeled_dialog)
        #     self_labeled_confidences.append(self_labeled_confidence)
            
        # print("self_labeled_dialogs", self_labeled_dialogs)