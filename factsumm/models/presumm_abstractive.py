from factsumm.models.abstractive import AbstractiveSummarizer
import torch
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from factsumm.models.helpers import StepCheckpointCallback
import argparse

# from abstractive import AbstractiveSummarizer
# from extractive import ExtractiveSummarizer
# from helpers import StepCheckpointCallback
import json

logger = logging.getLogger(__name__)


class PreSummAbstractive:
    def __init__(self, args):
        self.args = args

        self.model = AbstractiveSummarizer.load_from_checkpoint(
            "/home/toolkit/models/bert-base-uncased/epoch3.ckpt",
            # "/home/toolkit/models/bertsumextabs_cnndm_final_model/model_step_148000.pt",
            strict=False,
            hparams_file="configs/summarizer_v0.yaml",
        )

        if self.args["load_weights"]:
            # self.model.load_state_dict(torch.load(self.weights_path))
            # checkpoint = torch.load(self.weights_path)
            checkpoint = torch.load(
                # "/home/toolkit/models/bertsumextabs_cnndm_final_model/model_step_148000.pt"
                "/home/toolkit/models/bert-base-uncased/epoch3.ckpt"
            )
            # change key names from word_embedding_model to model
            for key in list(checkpoint["state_dict"]):
                if "word_embedding_model" in key:
                    checkpoint["state_dict"][
                        key.replace("word_embedding_model.", "")
                    ] = checkpoint["state_dict"][key]
                    del checkpoint["state_dict"][key]
            # for key in list(checkpoint["state_dict"]):
            #     if "model" in key:
            #         checkpoint["model"][key.replace("model.", "")] = checkpoint[
            #             "model"
            #         ][key]
            #         del checkpoint["model"][key]
            # self.model.load_state_dict(checkpoint["state_dict"])

            # Create learning rate logger
        lr_logger = LearningRateMonitor()
        self.args["callbacks"] = [lr_logger]

        if self.args["use_logger"] == "wandb":
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                log_model=(not args.no_wandb_logger_log_model),
            )
            self.args.logger = wandb_logger

        if self.args["use_custom_checkpoint_callback"]:
            try:
                self.args.checkpoint_callback = ModelCheckpoint(
                    save_top_k=-1,
                    every_n_epochs=1,
                    verbose=True,
                )
            except TypeError:
                logger.warning(
                    "'every_n_epochs' parameter of ModelCheckpoint is not found. "
                    + "Defaulting to its old name, 'period'."
                )
                self.args.checkpoint_callback = ModelCheckpoint(
                    save_top_k=-1,
                    period=1,
                    verbose=True,
                )

        if self.args["custom_checkpoint_every_n"]:
            custom_checkpoint_callback = StepCheckpointCallback(
                step_interval=args["custom_checkpoint_every_n"],
                save_path=args["weights_save_path"],
            )
            self.args["callbacks"].append(custom_checkpoint_callback)

        # if args["plugins"].__class__.__name__ == "DeepSpeedPlugin":
        #     self.args["plugins"] = args["plugins"]
        # elif args["plugins"] and args["plugins"].startswith("deepspeed"):
        #     deepspeed_config_path = args["plugins"].split(":")[1]
        #     with open(deepspeed_config_path, "r") as deepspeed_config_file:
        #         deepspeed_config = json.load(deepspeed_config_file)
        #     # args["plugins"] = DeepSpeedPlugin(config=deepspeed_config)
        #     args["plugins"] = DeepSpeedPlugin(stage=3)

        args["amp_backend"] = "apex"
        # args["amp_level"] = int("1")
        # If args is a dictionary, convert it to argparse.Namespace
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.trainer = Trainer.from_argparse_args(args)

    def train(self):
        self.trainer.fit(self.model)

    def test(self, text):
        return self.trainer.test(self.model)
