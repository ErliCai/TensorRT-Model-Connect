# SPDX-License-Identifier: Apache-2.0
"""Experimental native speech-token -> mel composition; no LLM or vocoder."""

from __future__ import annotations

from .flow_matching import solve_euler


class OfflineFlow:
    """Compose native conditioning and DiT with explicit per-request noise.

    B=1, offline/finalized inputs only. Prompt mel frames must correspond to
    prompt tokens at exactly two frames/token. This does not accept text.
    """

    def __init__(self, conditioner, estimator):
        if conditioner.device != estimator.device:
            raise ValueError("Conditioner and estimator must use the same device")
        self.conditioner, self.estimator = conditioner, estimator
        self.device = estimator.device

    def prepare(self, tokens, prompt_tokens, prompt_features, speaker):
        import torch

        for name, value in (("tokens", tokens), ("prompt_tokens", prompt_tokens)):
            if value.ndim != 2 or value.shape[0] != 1 or value.dtype != torch.int32 or value.device != self.device:
                raise ValueError(f"{name} must be INT32 [1, N] on {self.device}")
        if tokens.shape[1] == 0:
            raise ValueError("At least one target speech token is required")
        prompt_frames = prompt_tokens.shape[1] * 2
        if (prompt_features.shape != (1, prompt_frames, 80) or prompt_features.dtype != torch.float32
                or prompt_features.device != self.device or not torch.isfinite(prompt_features).all().item()):
            raise ValueError("prompt_features must be finite FP32 [1, 2*prompt_tokens, 80] on the engine device")
        frames = (tokens.shape[1] + prompt_tokens.shape[1]) * 2
        self.estimator.profile.validate_frames(frames)
        prepared = self.conditioner(torch.cat((prompt_tokens, tokens), dim=1), speaker)
        cond = torch.zeros((1, 80, frames), dtype=torch.float32, device=self.device)
        cond[:, :, :prompt_frames] = prompt_features.transpose(1, 2)
        prepared.update(cond=cond, mask=torch.ones((1, 1, frames), dtype=torch.float32, device=self.device))
        return prepared

    def __call__(self, tokens, prompt_tokens, prompt_features, speaker, noise):
        conditions = self.prepare(tokens, prompt_tokens, prompt_features, speaker)
        mel = solve_euler(self.estimator, **conditions, noise=noise)
        return mel[:, :, prompt_tokens.shape[1] * 2:].contiguous()
