# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments

from trl.trainer.dpo_config import FDivergenceType, FDivergenceConstants
from trl.trainer.utils import cap_exp

from ...extras.logging import get_logger

logger = get_logger(__name__)  


enable_debug = False


## NOTE(debug)
def check_for_nans(tensor, name):  
    if torch.isnan(tensor).any():  
        print(f"NaNs found in {name}")  
        return True  
    if torch.isinf(tensor).any():  
        print(f"Infs found in {name}")  
        return True  
    return False 


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.list_dpo_method = finetuning_args.list_dpo_method
        logger.info(f"list_dpo_method: {self.list_dpo_method}")
        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_middle_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        policy_middle_logits:  Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor","torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # print("use dpo loss")
            # losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            #     policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            # )
            losses, chosen_rewards, middle_rewards, rejected_rewards, logits_p1, logits_p2 = self.dpo_loss(
                policy_chosen_logps, policy_middle_logps, policy_rejected_logps, reference_chosen_logps, policy_middle_logits, reference_rejected_logps
            )

        return losses, chosen_rewards, middle_rewards, rejected_rewards, logits_p1, logits_p2 

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length
        batch_size = batch["input_ids"].size(0) // 3
        chosen_logps, middle_logps, rejected_logps = all_logps.split(batch_size, dim=0)  
        chosen_logits, middle_logits, rejected_logits = all_logits.split(batch_size, dim=0)  
        chosen_length, middle_length, rejected_length = valid_length.split(batch_size, dim=0) 

        if enable_debug:
            logger.info(f"batch inputs: {batch}")
            logger.info(f"all_logps: {all_logps}")
            logger.info(f"valid_length: {valid_length}")
            logger.info(f"chosen_logps: {chosen_logps}")
            logger.info(f"middle_logps: {middle_logps}")
            logger.info(f"rejected_logps: {rejected_logps}")
        # if check_for_nans(chosen_logps, "chosen_logps") or check_for_nans(middle_logps, "middle_logps") or check_for_nans(rejected_logps, "rejected_logps"):  
        #     print("NaNs found in concatenated_forward") 

        return chosen_logps, middle_logps, rejected_logps, chosen_logits, middle_logits, rejected_logits, chosen_logps / chosen_length  

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None, None  

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:  
            reference_chosen_logps, reference_middle_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)  
  
        return reference_chosen_logps, reference_middle_logps, reference_rejected_logps  

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}  
        (  
            policy_chosen_logps,  
            policy_middle_logps,  
            policy_rejected_logps,  
            policy_chosen_logits,  
            policy_middle_logits,  
            policy_rejected_logits,  
            policy_chosen_logps_avg,  
        ) = self.concatenated_forward(model, batch)  

        reference_chosen_logps, reference_middle_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)  
        losses, chosen_rewards, middle_rewards, rejected_rewards, logits_p1, logits_p2  = self.compute_preference_loss(  
            policy_chosen_logps,  
            policy_middle_logps,  
            policy_rejected_logps,  
            reference_chosen_logps,  
            reference_middle_logps,  
            reference_rejected_logps,  
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/middle".format(prefix)] = middle_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/middle".format(prefix)] = reference_middle_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/middle".format(prefix)] = policy_middle_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        metrics["{}logits/logits_p1".format(prefix)] = logits_p1.detach().mean().cpu()
        metrics["{}logits/logits_p2".format(prefix)] = logits_p2.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,  
        policy_middle_logps: torch.FloatTensor,  
        policy_rejected_logps: torch.FloatTensor,  
        reference_chosen_logps: torch.FloatTensor,  
        reference_middle_logps: torch.FloatTensor,  
        reference_rejected_logps: torch.FloatTensor,  
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:  
        """Compute the DPO loss for a batch of policy and reference model log probabilities using Plackett-Luce Model.  
    
        Args:  
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)  
            policy_middle_logps: Log probabilities of the policy model for the middle responses. Shape: (batch_size,)  
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)  
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)  
            reference_middle_logps: Log probabilities of the reference model for the middle responses. Shape: (batch_size,)  
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)  
    
        Returns:  
            A tuple of four tensors: (losses, chosen_rewards, middle_rewards, rejected_rewards).  
            The losses tensor contains the DPO loss for each example in the batch.  
            The chosen_rewards, middle_rewards, and rejected_rewards tensors contain the rewards for the chosen, middle, and rejected responses, respectively.  
        """ 
        # chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_chosen_logps.to(self.accelerator.device)
        # # middle_logratios = policy_middle_logps.to(self.accelerator.device) - (not self.reference_free) * reference_middle_logps.to(self.accelerator.device)  
        # middle_logratios = policy_middle_logps.to(self.accelerator.device) - (  
        #     not self.reference_free  
        # ) * reference_middle_logps.to(self.accelerator.device) 
        # rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
        #     not self.reference_free
        # ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            # pi_logratios = policy_chosen_logps - policy_rejected_logps
            # if self.reference_free:
            #     ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            # else:
            #     ref_logratios = reference_chosen_logps - reference_rejected_logps

            chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            middle_logratios = policy_middle_logps.to(self.accelerator.device) - reference_middle_logps.to(self.accelerator.device)
            rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)
            if enable_debug:
                logger.info(f"chosen_logratios: {chosen_logratios}")
                logger.info(f"policy_chosen_logps: {policy_chosen_logps}")
                logger.info(f"reference_chosen_logps: {reference_chosen_logps}")            

                logger.info(f"middle_logratios: {middle_logratios}")
                logger.info(f"policy_middle_logps: {policy_middle_logps}")
                logger.info(f"reference_middle_logps: {reference_middle_logps}")

                logger.info(f"rejected_logratios: {rejected_logratios}")
                logger.info(f"policy_rejected_logps: {policy_rejected_logps}")
                logger.info(f"reference_rejected_logps: {reference_rejected_logps}")
            # Ensure no negative values before taking exp  
            # chosen_logratios = torch.clamp(chosen_logratios, min=-1000, max=100)  
            # middle_logratios = torch.clamp(middle_logratios, min=-1000, max=100)  
            # rejected_logratios = torch.clamp(rejected_logratios, min=-1000, max=100)  
            # pi_logratios = policy_middle_logps - policy_rejected_logps
            # ref_logratios = reference_middle_logps - reference_rejected_logps
        if self.list_dpo_method == "test":
            chosen_logratios = chosen_logratios.to(self.accelerator.device)
            # middle_logratios = middle_logratios.to(self.accelerator.device)
            rejected_logratios = rejected_logratios.to(self.accelerator.device)

            # pi_logratios = pi_logratios.to(self.accelerator.device)
            # ref_logratios = ref_logratios.to(self.accelerator.device)

            # part one
            r1 = torch.exp(self.beta * chosen_logratios)
            # r2 = torch.exp(self.beta * middle_logratios)
            r3 = torch.exp(self.beta * rejected_logratios)

            r1 = r1.to(self.accelerator.device)
            # r2 = r2.to(self.accelerator.device)
            r3 = r3.to(self.accelerator.device)
            # Avoid division by zero  
            # denom = r1 + r2 + r3  
            # denom = torch.clamp(denom, min=1e-10) 
            # p1  = r1 / denom 
            # p1 = torch.clamp(p1, min=1e-3)
            # p1  = r1 / (r1 + r2 + r3) 
            # logits_p1 = torch.log(p1)
            # part two
            # p2 = pi_logratios - ref_logratios
            # logits_p2 = F.logsigmoid(self.beta * p2)
            # p2 = r2 / (r2 + r3)
            # logits_p2 = torch.log(p2)
            # p2 = r2 / torch.clamp(r2 + r3, min=1e-10)  
            # p2 = torch.clamp(p2, min=1e-3)
            # logits_p2 = torch.log(p2)  
            # losses = -(logits_p1 + logits_p2)
            # losses = -logits_p2
            # losses = -logits_p1
            p3 = r1 / (r1 + r3)
            logits_p3 = torch.log(p3)
            logits_p1 = torch.tensor([-0.0222, -2.0782, -0.6932, -0.5227]).to(self.accelerator.device) 
            logits_p2 = torch.tensor([-0.0222, -2.0782, -0.6932, -0.5227]).to(self.accelerator.device)   
            losses = -logits_p3
            if enable_debug:
                # logger.info(f"logits_p1: {logits_p1}, logits_p2:{logits_p2}")
                logger.info(f"r1: {r1}")
                # logger.info(f"r2: {r2}")
                logger.info(f"r3: {r3}")
                logger.info(f"p3: {p3}")
                logger.info(f"losses: {losses}")

        elif self.list_dpo_method == "v1":
            chosen_logratios = chosen_logratios.to(self.accelerator.device)
            middle_logratios = middle_logratios.to(self.accelerator.device)
            rejected_logratios = rejected_logratios.to(self.accelerator.device)

            # pi_logratios = pi_logratios.to(self.accelerator.device)
            # ref_logratios = ref_logratios.to(self.accelerator.device)

            # part one
            r1 = torch.exp(self.beta * chosen_logratios)
            r2 = torch.exp(self.beta * middle_logratios)
            r3 = torch.exp(self.beta * rejected_logratios)

            r1 = r1.to(self.accelerator.device)
            r2 = r2.to(self.accelerator.device)
            r3 = r3.to(self.accelerator.device)
            # Avoid division by zero  
            denom = r1 + r2 + r3  
            denom = torch.clamp(denom, min=1e-10) 
            p1  = r1 / denom 
            p1 = torch.clamp(p1, min=1e-3)
            # p1  = r1 / (r1 + r2 + r3) 
            logits_p1 = torch.log(p1)
            # part two
            # p2 = pi_logratios - ref_logratios
            # logits_p2 = F.logsigmoid(self.beta * p2)
            p2 = r2 / (r2 + r3)
            # logits_p2 = torch.log(p2)
            p2 = r2 / torch.clamp(r2 + r3, min=1e-10)  
            # p2 = torch.clamp(p2, min=1e-3)
            logits_p2 = torch.log(p2)  
            losses = -(logits_p1 + logits_p2)
            if enable_debug:
                # logger.info(f"logits_p1: {logits_p1}, logits_p2:{logits_p2}")
                logger.info(f"r1: {r1}")
                logger.info(f"r2: {r2}")
                logger.info(f"r3: {r3}")
                logger.info(f"p1: {p1}")
                logger.info(f"p2: {p2}")
                logger.info(f"logits_p1:{logits_p1}")
                logger.info(f"logits_p2:{logits_p2}")
                logger.info(f"losses: {losses}")

        elif self.list_dpo_method == "v2":
            chosen_logratios = chosen_logratios.to(self.accelerator.device)
            middle_logratios = middle_logratios.to(self.accelerator.device)
            rejected_logratios = rejected_logratios.to(self.accelerator.device)


            # part one
            r1 = torch.exp(self.beta * chosen_logratios)
            r2 = torch.exp(self.beta * middle_logratios)
            r3 = torch.exp(self.beta * rejected_logratios)

            r1 = r1.to(self.accelerator.device)
            r2 = r2.to(self.accelerator.device)
            r3 = r3.to(self.accelerator.device)
            # Avoid division by zero  
            denom = r1 + r2 + r3  
            # denom = torch.clamp(denom, min=1e-10) 
            p1  = r1 / denom 
            # p1 = torch.clamp(p1, min=1e-3)
            # p1  = r1 / (r1 + r2 + r3) 
            logits_p1 = torch.log(p1)
            # part two
            # p2 = pi_logratios - ref_logratios
            # logits_p2 = F.logsigmoid(self.beta * p2)
            p2 = r2 / (r2 + r3)

            logits_p2 = torch.log(p2)  
            losses = -(logits_p1 + logits_p2)
            if enable_debug:
                # logger.info(f"logits_p1: {logits_p1}, logits_p2:{logits_p2}")
                logger.info(f"r1: {r1}")
                logger.info(f"r2: {r2}")
                logger.info(f"r3: {r3}")
                logger.info(f"p1: {p1}")
                logger.info(f"p2: {p2}")
                logger.info(f"logits_p1:{logits_p1}")
                logger.info(f"logits_p2:{logits_p2}")
                logger.info(f"losses: {losses}")

        elif self.list_dpo_method == "v3":
            chosen_logratios = chosen_logratios.to(self.accelerator.device)
            middle_logratios = middle_logratios.to(self.accelerator.device)
            rejected_logratios = rejected_logratios.to(self.accelerator.device)


            # part one
            r1 = torch.exp(self.beta * chosen_logratios)
            r2 = torch.exp(self.beta * middle_logratios)
            r3 = torch.exp(self.beta * rejected_logratios)

            r1 = r1.to(self.accelerator.device)
            r2 = r2.to(self.accelerator.device)
            r3 = r3.to(self.accelerator.device)
            # Avoid division by zero  
            denom = r1 + r2 + r3  
            # denom = torch.clamp(denom, min=1e-10) 
            p1  = r1 / denom 
            # p1 = torch.clamp(p1, min=1e-3)
            # p1  = r1 / (r1 + r2 + r3) 
            logits_p1 = torch.log(p1)
            # part two
            # p2 = pi_logratios - ref_logratios
            # logits_p2 = F.logsigmoid(self.beta * p2)
            logits_p2 = torch.tensor([-0.0222, -2.0782, -0.6932, -0.5227]).to(self.accelerator.device) 
            losses = -logits_p1
            # p2 = r2 / (r2 + r3)

            # logits_p2 = torch.log(p2)  
            # losses = -(logits_p1 + logits_p2)
            if enable_debug:
                # logger.info(f"logits_p1: {logits_p1}, logits_p2:{logits_p2}")
                logger.info(f"r1: {r1}")
                logger.info(f"r2: {r2}")
                logger.info(f"r3: {r3}")
                logger.info(f"p1: {p1}")
                # logger.info(f"p2: {p2}")
                logger.info(f"logits_p1:{logits_p1}")
                # logger.info(f"logits_p2:{logits_p2}")
                logger.info(f"losses: {losses}")

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        middle_rewards = (  
            self.beta  
            * (
                policy_middle_logps.to(self.accelerator.device) - reference_middle_logps.to(self.accelerator.device)
            ).detach()  
        )  
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )
        return losses, chosen_rewards, middle_rewards, rejected_rewards, logits_p1, logits_p2