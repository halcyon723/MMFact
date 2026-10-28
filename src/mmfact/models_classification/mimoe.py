"""MIMoE-Corrected: interaction-supervised sparse fusion over refined tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from mmfact.base import AbstractDetector
from mmfact.materials.encoders import material_encoder
from mmfact.models_classification._encoders import (
    BertEncoder,
    CLIPImageEncoder,
    CLIPTextEncoder,
    MAETokenEncoder,
)
from mmfact.models_classification._transformers import ViTExpert, sinusoidal_positions
from mmfact.registry import register_model


def _mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class _TokenAttention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.scores = _mlp(dimension, dimension, 1)

    def forward(self, values):
        # The release uses signed attention scores without token-wise softmax.
        return (self.scores(values) * values).sum(1)


class _Mixture(nn.Module):
    """iMoE: attention-guided dispatch, independent ViT experts and CLS pooling."""

    def __init__(self, dimension, experts, heads, multiplier):
        super().__init__()
        self.attention = _TokenAttention(dimension)
        self.gate = _mlp(dimension, dimension, experts)
        self.experts = nn.ModuleList(
            ViTExpert(dimension, heads, multiplier) for _ in range(experts)
        )

    def forward(self, values):
        dispatch = self.gate(self.attention(values))
        positioned = values + sinusoidal_positions(values)
        features = torch.stack([expert(positioned)[:, 0] for expert in self.experts], 1)
        return (dispatch.unsqueeze(-1) * features).sum(1)


def _js_divergence(first, second):
    """JS between Bernoulli predictions, computed with stable log probabilities."""
    log_first = torch.stack((F.logsigmoid(-first), F.logsigmoid(first)), -1)
    log_second = torch.stack((F.logsigmoid(-second), F.logsigmoid(second)), -1)
    mean = torch.logaddexp(log_first, log_second) - math.log(2)
    return (
        0.5
        * (
            (log_first.exp() * (log_first - mean)).sum(-1)
            + (log_second.exp() * (log_second - mean)).sum(-1)
        )
    ).clamp_min(0)


class _Interaction(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.attention = _TokenAttention(dimension)
        self.gate = _mlp(dimension, dimension, 4)

    def forward(self, text, image, clip_text, clip_image):
        return self.gate(self.attention(torch.stack((text, image, clip_text, clip_image), 1)))


class _AdaptiveNorm(nn.Module):
    def __init__(self, dimension, epsilon):
        super().__init__()
        self.mean = _mlp(1, dimension, 1)
        self.scale = _mlp(1, dimension, 1)
        self.epsilon = epsilon

    def forward(self, features, logit):
        probability = logit.detach().sigmoid().unsqueeze(-1)
        centered = features - features.mean(-1, keepdim=True)
        normalized = centered * (centered.square().mean(-1, keepdim=True) + self.epsilon).rsqrt()
        return self.scale(probability) * normalized + self.mean(probability)


@register_model("MIMoE")
class MIMoE(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        p = config["model"]["parameters"]
        integer_keys = (
            "routing_dim",
            "hidden_dim",
            "clip_hidden_dim",
            "num_experts",
            "num_heads",
            "ffn_multiplier",
        )
        if any(type(p[key]) is not int or p[key] < 1 for key in integer_keys):
            raise ValueError("MIMoE dimensions, heads and expert counts must be positive integers.")
        if p["routing_dim"] % p["num_heads"]:
            raise ValueError("MIMoE routing_dim must be divisible by num_heads.")
        scalar_keys = (
            "agreement_threshold",
            "semantic_threshold",
            "alpha",
            "beta",
            "router_z_weight",
            "balance_weight",
            "epsilon",
            "adain_epsilon",
        )
        if (
            not all(type(p[key]) in (int, float) and math.isfinite(p[key]) for key in scalar_keys)
            or not 0 <= p["agreement_threshold"] <= math.log(2)
            or not -1 <= p["semantic_threshold"] <= 1
            or min(p[key] for key in ("alpha", "beta", "router_z_weight", "balance_weight")) < 0
            or min(p[key] for key in ("epsilon", "adain_epsilon")) <= 0
        ):
            raise ValueError("MIMoE thresholds, loss weights and numerical settings are invalid.")
        positive_weight = p["positive_weight"]
        if (
            type(positive_weight) not in (int, float)
            or not math.isfinite(positive_weight)
            or positive_weight <= 0
        ):
            raise ValueError("MIMoE positive_weight must be a finite positive number.")
        if (
            type(p["text"]["max_length"]) is not int
            or p["text"]["max_length"] < 2
            or type(p["image"]["image_size"]) is not int
            or p["image"]["image_size"] < 1
            or type(p["image"]["minimum_size"]) is not int
            or p["image"]["minimum_size"] < 0
            or any(
                type(p[key]["freeze"]) is not bool
                for key in ("text", "image", "clip_text", "clip_image")
            )
            or type(p["clip_text"]["max_length"]) is not int
            or p["clip_text"]["max_length"] < 2
            or p["image"]["augment"] is not False
            or p["clip_text"]["model_name"] != p["clip_image"]["model_name"]
            or p["clip_text"]["revision"] != p["clip_image"]["revision"]
            or p["clip_text"]["normalize"] is not False
            or p["clip_image"]["normalize"] is not False
        ):
            raise ValueError(
                "MIMoE requires valid token sizes, fixed MAE inputs and matching raw CLIP towers."
            )
        self.text_encoder = material_encoder(
            config, "text_encoder", BertEncoder, **p["text"], output="tokens", mask_padding=False
        )
        self.image_encoder = material_encoder(
            config, "image_encoder", MAETokenEncoder, **p["image"]
        )
        self.clip_text_encoder = material_encoder(
            config, "clip_text_encoder", CLIPTextEncoder, **p["clip_text"]
        )
        self.clip_image_encoder = material_encoder(
            config, "clip_image_encoder", CLIPImageEncoder, **p["clip_image"]
        )
        dimension, hidden = p["routing_dim"], p["hidden_dim"]
        if (
            self.text_encoder.output_dim != dimension
            or self.image_encoder.output_dim != dimension
            or self.clip_text_encoder.output_dim != self.clip_image_encoder.output_dim
        ):
            raise ValueError("MIMoE requires matching BERT/MAE and CLIP representation dimensions.")

        def mixture():
            return _Mixture(dimension, p["num_experts"], p["num_heads"], p["ffn_multiplier"])

        self.refine = nn.ModuleDict({key: mixture() for key in ("text", "image", "multimodal")})
        self.trim = nn.ModuleDict(
            {
                key: nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU())
                for key in ("text", "image")
            }
        )
        self.unimodal_heads = nn.ModuleDict({key: nn.Linear(hidden, 1) for key in self.trim})
        self.clip_projection = nn.ModuleDict(
            {
                key: _mlp(self.clip_text_encoder.output_dim, p["clip_hidden_dim"], hidden)
                for key in ("text", "image")
            }
        )
        self.interaction = _Interaction(hidden)
        self.normalization = nn.ModuleDict(
            {key: _AdaptiveNorm(dimension, p["adain_epsilon"]) for key in self.trim}
        )
        self.fusion = nn.ModuleList(mixture() for _ in range(4))
        self.classifier = _mlp(dimension, hidden, 1)
        self.alpha, self.beta = p["alpha"], p["beta"]
        self.router_z_weight, self.balance_weight = p["router_z_weight"], p["balance_weight"]
        self.agreement_threshold, self.semantic_threshold = (
            p["agreement_threshold"],
            p["semantic_threshold"],
        )
        self.routing = p.get("routing", "hard")
        if self.routing not in ("hard", "soft"):
            raise ValueError("MIMoE routing must be 'hard' or 'soft'.")
        self.epsilon = p["epsilon"]
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))

    def optimizer_parameter_groups(self):
        routing = {id(parameter) for parameter in self.interaction.parameters()}
        return {
            "default": [p for p in self.parameters() if p.requires_grad and id(p) not in routing],
            "interaction": [p for p in self.interaction.parameters() if p.requires_grad],
        }

    def forward(self, batch):
        text = self.text_encoder(batch["text"])
        image = self.image_encoder(batch["image_path"])
        features = {
            "text": self.refine["text"](text),
            "image": self.refine["image"](image),
            "multimodal": self.refine["multimodal"](torch.cat((text, image), 1)),
        }
        compact = {key: self.trim[key](features[key]) for key in self.trim}
        unimodal = {key: self.unimodal_heads[key](compact[key]).squeeze(-1) for key in self.trim}
        clip_text = self.clip_text_encoder(batch["text"])
        clip_image = self.clip_image_encoder(batch["image_path"])
        gate = self.interaction(
            compact["text"],
            compact["image"],
            self.clip_projection["text"](clip_text),
            self.clip_projection["image"](clip_image),
        )
        tokens = torch.stack(
            (
                self.normalization["text"](features["text"], unimodal["text"]),
                features["multimodal"],
                self.normalization["image"](features["image"], unimodal["image"]),
            ),
            1,
        )
        dispatch = gate.softmax(-1)
        if self.routing == "soft":
            fused = sum(
                dispatch[:, index].unsqueeze(-1) * expert(tokens)
                for index, expert in enumerate(self.fusion)
            )
        else:
            selected = dispatch.argmax(-1)
            fused = torch.zeros_like(features["multimodal"])
            for index, expert in enumerate(self.fusion):
                mask = selected == index
                if mask.any():
                    fused[mask] = expert(tokens[mask])
        return {
            "final": self.classifier(fused).squeeze(-1),
            "text": unimodal["text"],
            "image": unimodal["image"],
            "gate": gate,
            "clip_text": clip_text,
            "clip_image": clip_image,
        }

    def calculate_loss(self, batch):
        outputs = self(batch)
        labels = batch["labels"].float()
        with torch.no_grad():
            agreement = _js_divergence(outputs["text"], outputs["image"]) < self.agreement_threshold
            alignment = (
                F.cosine_similarity(outputs["clip_text"], outputs["clip_image"], eps=self.epsilon)
                > self.semantic_threshold
            )
            targets = 2 * agreement.long() + alignment.long()
        gate = outputs["gate"]
        distribution = gate.softmax(-1).mean(0)
        return {
            "classification": F.binary_cross_entropy_with_logits(
                outputs["final"], labels, pos_weight=self.positive_weight
            ),
            "unimodal": self.alpha
            * sum(
                F.binary_cross_entropy_with_logits(
                    outputs[key], labels, pos_weight=self.positive_weight
                )
                for key in ("text", "image")
            )
            / 2,
            "interaction": self.beta * F.cross_entropy(gate, targets),
            "router_z": self.beta * self.router_z_weight * gate.logsumexp(-1).square().mean(),
            "balance": self.beta * self.balance_weight * (distribution - 0.25).square().mean(),
        }

    def predict(self, batch):
        mode = self.training
        self.eval()
        try:
            logits = self(batch)["final"]
            return torch.stack((torch.zeros_like(logits), logits), -1)
        finally:
            self.train(mode)
