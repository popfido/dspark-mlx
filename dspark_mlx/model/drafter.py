# Copyright 2026 popfido
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DeepSeek DSpark (DeepSeek-V4-Flash/Pro-DSpark, deepseek-ai/DeepSpec)

"""DSparkDrafter: the full draft stack (``inference/model.py`` Transformer.forward_spec).

Owns the shared token embedding + LM head and the ``n_mtp_layers`` DSparkBlocks. The first
block projects the main hidden and embeds the draft block; the last produces the draft
tokens, their (Markov-biased) logits, and per-token confidence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .block import DSparkBlock
from .config import DSparkArgs


class DSparkDrafter(nn.Module):
    def __init__(self, args: DSparkArgs, max_seq_len: int = 8192):
        super().__init__()
        self.n_mtp_layers = args.n_mtp_layers
        self.block_size = args.dspark_block_size
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.blocks = [DSparkBlock(args, i, max_seq_len) for i in range(args.n_mtp_layers)]

    def forward_spec(
        self, input_ids: mx.array, main_hidden: mx.array, start_pos: int = 0
    ) -> Optional[Tuple[mx.array, mx.array, mx.array]]:
        """Prefill (start_pos==0) seeds window KV and returns None; decode drafts a block."""
        h, main_x = self.blocks[0].forward_embed(main_hidden, input_ids, self.embed)
        for blk in self.blocks:
            h = blk(h, start_pos, input_ids, main_x)
        if start_pos == 0:
            return None
        return self.blocks[-1].forward_head(h, input_ids, self.head)

    def advance(self, main_hidden: mx.array, position: int) -> None:
        """Slide every block's window over one committed token at ``position``.

        ``main_hidden`` is the base hidden ([b, D] or [b, 1, D]); projected once via the
        stage-0 main_proj/main_norm and fed to each block. The generate loop calls this for
        each token accepted within a block (forward_spec only advances the anchor).
        """
        mh = main_hidden.reshape(main_hidden.shape[0], -1)
        main_x = self.blocks[0].main_norm(self.blocks[0].main_proj(mh))
        for blk in self.blocks:
            blk.advance(main_x, position)
