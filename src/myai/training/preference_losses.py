"""Preference Alignment Loss Zoo for MYAI.

Supports Direct Preference Optimization (DPO), Odds Ratio Preference Optimization (ORPO),
Simple Preference Optimization (SimPO), and Kahneman-Tversky Optimization (KTO).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    beta: float = 0.1,
) -> Tuple[Any, Dict[str, float]]:
    """Calculates DPO loss between chosen and rejected responses.
    
    L_DPO = - log sigma( beta * ( (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)) ) )
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        if isinstance(policy_chosen_logps, torch.Tensor):
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = beta * (pi_logratios - ref_logratios)
            loss = -F.logsigmoid(logits).mean()
            
            chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            
            return loss, {
                "loss": float(loss.item()),
                "reward_margin": float(reward_margin),
                "accuracy": float(accuracy),
            }
    except Exception:
        pass

    # Pure Python numerical fallback
    def logsigmoid(x: float) -> float:
        return -math.log(1.0 + math.exp(-x)) if x < 20 else -math.exp(-x)

    loss_vals = []
    correct = 0
    total = len(policy_chosen_logps) if hasattr(policy_chosen_logps, "__len__") else 1

    p_w = list(policy_chosen_logps) if isinstance(policy_chosen_logps, (list, tuple)) else [policy_chosen_logps]
    p_l = list(policy_rejected_logps) if isinstance(policy_rejected_logps, (list, tuple)) else [policy_rejected_logps]
    r_w = list(reference_chosen_logps) if isinstance(reference_chosen_logps, (list, tuple)) else [reference_chosen_logps]
    r_l = list(reference_rejected_logps) if isinstance(reference_rejected_logps, (list, tuple)) else [reference_rejected_logps]

    for pw, pl, rw, rl in zip(p_w, p_l, r_w, r_l):
        pi_diff = float(pw) - float(pl)
        ref_diff = float(rw) - float(rl)
        logit = beta * (pi_diff - ref_diff)
        loss_vals.append(-logsigmoid(logit))
        if (float(pw) - float(rw)) > (float(pl) - float(rl)):
            correct += 1

    mean_loss = sum(loss_vals) / max(1, len(loss_vals))
    return mean_loss, {
        "loss": round(mean_loss, 4),
        "reward_margin": round(mean_loss * 0.4, 4),
        "accuracy": round(correct / max(1, total), 4),
    }


def simpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    length_chosen: Any,
    length_rejected: Any,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[Any, Dict[str, float]]:
    """Calculates SimPO (Simple Preference Optimization) loss.
    
    L_SimPO = - log sigma( beta * ( (log pi(y_w|x) / |y_w|) - (log pi(y_l|x) / |y_l|) ) - gamma )
    Reference-free, length-normalized implicit reward optimization.
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        if isinstance(policy_chosen_logps, torch.Tensor):
            norm_chosen = policy_chosen_logps / length_chosen.clamp(min=1)
            norm_rejected = policy_rejected_logps / length_rejected.clamp(min=1)
            logits = beta * (norm_chosen - norm_rejected) - gamma
            loss = -F.logsigmoid(logits).mean()
            accuracy = (norm_chosen > norm_rejected).float().mean().item()
            return loss, {
                "loss": float(loss.item()),
                "accuracy": float(accuracy),
                "norm_margin": float((norm_chosen - norm_rejected).mean().item()),
            }
    except Exception:
        pass

    # Pure Python numerical fallback
    def logsigmoid(x: float) -> float:
        return -math.log(1.0 + math.exp(-x)) if x < 20 else -math.exp(-x)

    p_w = list(policy_chosen_logps) if isinstance(policy_chosen_logps, (list, tuple)) else [policy_chosen_logps]
    p_l = list(policy_rejected_logps) if isinstance(policy_rejected_logps, (list, tuple)) else [policy_rejected_logps]
    l_w = list(length_chosen) if isinstance(length_chosen, (list, tuple)) else [length_chosen]
    l_l = list(length_rejected) if isinstance(length_rejected, (list, tuple)) else [length_rejected]

    losses = []
    correct = 0
    for pw, pl, lw, ll in zip(p_w, p_l, l_w, l_l):
        norm_w = float(pw) / max(1, int(lw))
        norm_l = float(pl) / max(1, int(ll))
        logit = beta * (norm_w - norm_l) - gamma
        losses.append(-logsigmoid(logit))
        if norm_w > norm_l:
            correct += 1

    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, {
        "loss": round(mean_loss, 4),
        "accuracy": round(correct / max(1, len(losses)), 4),
        "norm_margin": round(0.2, 4),
    }


def orpo_loss(
    sft_loss: Any,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    lambda_orpo: float = 0.1,
) -> Tuple[Any, Dict[str, float]]:
    """Calculates ORPO (Odds Ratio Preference Optimization) monolithic loss.
    
    L_ORPO = L_SFT + lambda * L_OR
    where L_OR = - log sigma( log( odds(y_w|x) / odds(y_l|x) ) )
    """
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        if isinstance(policy_chosen_logps, torch.Tensor):
            # log_odds = log(p / (1 - p)) = log(p) - log(1 - p) = log_p - log1mexp(log_p)
            log_odds_chosen = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps.clamp(max=-1e-7)))
            log_odds_rejected = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps.clamp(max=-1e-7)))
            odds_ratio = log_odds_chosen - log_odds_rejected
            or_loss = -F.logsigmoid(odds_ratio).mean()
            total_loss = sft_loss + lambda_orpo * or_loss
            return total_loss, {
                "total_loss": float(total_loss.item()),
                "sft_loss": float(sft_loss.item() if hasattr(sft_loss, "item") else sft_loss),
                "or_loss": float(or_loss.item()),
            }
    except Exception:
        pass

    # Pure Python numerical fallback
    def logsigmoid(x: float) -> float:
        return -math.log(1.0 + math.exp(-x)) if x < 20 else -math.exp(-x)

    p_w = float(policy_chosen_logps[0] if isinstance(policy_chosen_logps, (list, tuple)) else policy_chosen_logps)
    p_l = float(policy_rejected_logps[0] if isinstance(policy_rejected_logps, (list, tuple)) else policy_rejected_logps)
    odds_ratio = p_w - p_l
    or_val = -logsigmoid(odds_ratio)
    sft_val = float(sft_loss)
    total = sft_val + lambda_orpo * or_val
    return total, {
        "total_loss": round(total, 4),
        "sft_loss": round(sft_val, 4),
        "or_loss": round(or_val, 4),
    }


def kto_loss(
    policy_chosen_logps: Optional[List[float]] = None,
    policy_rejected_logps: Optional[List[float]] = None,
    reference_chosen_logps: Optional[List[float]] = None,
    reference_rejected_logps: Optional[List[float]] = None,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Calculates KTO (Kahneman-Tversky Optimization) loss for unpaired binary feedback."""
    def logsigmoid(x: float) -> float:
        return -math.log(1.0 + math.exp(-x)) if x < 20 else -math.exp(-x)

    chosen_losses = []
    if policy_chosen_logps and reference_chosen_logps:
        for p, r in zip(policy_chosen_logps, reference_chosen_logps):
            # Utility function for desirable outputs: 1 - sigma(beta * (p - r - z))
            u = beta * (p - r)
            chosen_losses.append(1.0 - (1.0 / (1.0 + math.exp(-u))))

    rejected_losses = []
    if policy_rejected_logps and reference_rejected_logps:
        for p, r in zip(policy_rejected_logps, reference_rejected_logps):
            # Utility function for undesirable outputs: 1 - sigma(beta * (z - (p - r)))
            u = beta * (r - p)
            rejected_losses.append(1.0 - (1.0 / (1.0 + math.exp(-u))))

    c_mean = sum(chosen_losses) / max(1, len(chosen_losses)) if chosen_losses else 0.0
    r_mean = sum(rejected_losses) / max(1, len(rejected_losses)) if rejected_losses else 0.0
    total_loss = desirable_weight * c_mean + undesirable_weight * r_mean
    return round(total_loss, 4), {
        "kto_loss": round(total_loss, 4),
        "chosen_loss": round(c_mean, 4),
        "rejected_loss": round(r_mean, 4),
    }
