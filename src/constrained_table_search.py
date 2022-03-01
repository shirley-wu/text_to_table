import copy
import logging
import math
from typing import Optional

import torch
from fairseq.search import Search
from torch import Tensor

from .table_state import SpecialTokens, TableState

logger = logging.getLogger("fairseq_cli.train")

class ConstrainedTableBeamSearch(Search):
    def __init__(self, tgt_dict, special_tokens: SpecialTokens, table_max_columns: int):
        super().__init__(tgt_dict)
        self.special_tokens = special_tokens
        self.table_max_columns = table_max_columns
        self.supports_constraints = True
        self.constraint_states = None
        self.stop_on_max_len = False  # NOTE

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        bsz = batch_constraints.shape[0]
        self.constraint_states = [
            [TableState(self.special_tokens, self.table_max_columns) for _ in range(beam_size)] for _ in range(bsz)
        ]

    @torch.jit.export
    def prune_sentences(self, batch_idxs: Tensor):
        self.constraint_states = [
            self.constraint_states[i] for i in batch_idxs.tolist()
        ]

    @torch.jit.export
    def update_constraints(self, active_hypos: Tensor):
        if self.constraint_states:
            batch_size = active_hypos.size(0)
            for sentid in range(batch_size):
                self.constraint_states[sentid] = [
                    copy.deepcopy(self.constraint_states[sentid][i]) for i in active_hypos[sentid]
                ]

    @torch.jit.export
    def step(
            self,
            step: int,
            lprobs: Tensor,
            scores: Optional[Tensor],
            prev_output_tokens: Optional[Tensor] = None,
            original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        # update state
        batch_ban_mask = torch.zeros((bsz * beam_size, vocab_size), dtype=torch.bool, device=lprobs.device)
        for i in range(bsz):
            for j in range(beam_size):
                lprob_id = i * beam_size + j
                self.constraint_states[i][j].step(prev_output_tokens[lprob_id, -1])
                # enforce
                enforce_tokens = self.constraint_states[i][j].enforce_tokens
                if len(enforce_tokens) > 0:
                    batch_ban_mask[lprob_id] = 1
                    batch_ban_mask[lprob_id, torch.LongTensor(enforce_tokens).to(lprobs.device)] = 0
                # ban
                ban_tokens = self.constraint_states[i][j].ban_tokens
                if len(ban_tokens) > 0:
                    batch_ban_mask[lprob_id, torch.LongTensor(ban_tokens).to(lprobs.device)] = 1

        # update lprobs
        if batch_ban_mask.sum() > 0:
            assert torch.all(batch_ban_mask.min(dim=1)[0] == 0)
            lprobs = lprobs.view(-1, vocab_size)
            reach_max_length = torch.isinf(
                torch.cat([lprobs[:, :self.eos], lprobs[:, self.eos + 1:], ], dim=1)
            ).all(dim=1)
            if torch.any(batch_ban_mask[reach_max_length, self.eos]):
                logger.warning("Decoding reach max length so forced to stop. However wrong format.")
            batch_ban_mask[reach_max_length, self.eos] = False
            lprobs[batch_ban_mask] = -math.inf
            lprobs = lprobs.view(bsz, beam_size, vocab_size)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        # update constrained states
        self.constraint_states = [[copy.deepcopy(self.constraint_states[i][j]) for j in beams_buf[i]] for i in range(bsz)]

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        return scores_buf, indices_buf, beams_buf
