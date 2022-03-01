import copy
import logging
from typing import Any, Dict, Optional
from typing import List

import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import (
    BARTModel,
    bart_base_architecture,
    bart_large_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerDecoder,
)
from torch import Tensor

from src.modules.transformer_layer import TransformerRelativeEmbeddingsDecoderLayer
from src.table_state import TableState, get_batch_relative_column_ids

logger = logging.getLogger(__name__)


@register_model("bart_ours")
class BARTOurs(BARTModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        assert args.return_relative_column_strs is not None
        return TransformerOursDecoder(
            args, tgt_dict, embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            features_only=False,
            classification_head_name=None,
            token_embeddings=None,
            prev_output_relative_column_ids: Optional[Tensor] = None,
            **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            prev_output_relative_column_ids=prev_output_relative_column_ids,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[src_tokens.eq(self.encoder.dictionary.eos()), :].view(
                x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra

    def set_special_tokens(self, special_tokens):
        self.decoder.special_tokens = special_tokens


class TransformerOursDecoder(TransformerDecoder):
    # almost completely copied from TransformerDecoder, except the embedding layer (marked by [ours])

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.warned_empty_relative_column_ids_once = False
        self.token_padding_idx = dictionary.pad()
        self.special_tokens = None

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if self.args.return_relative_column_strs is not None:
            return TransformerRelativeEmbeddingsDecoderLayer(args, no_encoder_attn)
        else:
            return super().build_decoder_layer(args, no_encoder_attn=no_encoder_attn)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            prev_output_relative_column_ids: Optional[Tensor] = None,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            prev_output_relative_column_ids=prev_output_relative_column_ids,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            prev_output_relative_column_ids: Optional[Tensor] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            prev_output_relative_column_ids=prev_output_relative_column_ids,
        )

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            prev_output_relative_column_ids: Optional[Tensor] = None,
    ):
        prev_output_relative_column_ids = self.get_prev_output_relative_column_ids(
            prev_output_relative_column_ids, prev_output_tokens, incremental_state
        )

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if self.args.return_relative_column_strs is not None:
                x, layer_attn, _ = layer(
                    x,
                    prev_output_relative_column_ids,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def get_prev_output_relative_column_ids(self, x_relation, x, incremental_state):
        bsz, x_length = x.shape

        if x_relation is None:
            if not self.warned_empty_relative_column_ids_once:
                logger.warning("You use empty column ids")
                self.warned_empty_relative_column_ids_once = True
            assert self.special_tokens is not None

            if incremental_state is not None:
                prev_output_tokens = x[:, -1:]

                # use incremental states. Save TableState
                table_states = self._get_input_buffer(incremental_state)
                # init state
                if table_states is None:
                    table_states = [TableState(self.special_tokens,
                                               return_relative_column_strs=self.args.return_relative_column_strs)
                                    for _ in range(len(prev_output_tokens))]
                # state step
                for i in range(bsz):
                    table_states[i].step(prev_output_tokens[i, -1])
                # get column ids
                x_relation = torch.stack(
                    [table_states[i].relative_column_ids for i in range(bsz)], dim=0
                ).to(prev_output_tokens.device).unsqueeze(1)  # (bsz, 1, x_length)

                # set incremental states
                self._set_input_buffer(incremental_state, table_states)
            else:
                x_relation = get_batch_relative_column_ids(
                    x, self.special_tokens, self.args.return_relative_column_strs,
                    token_padding_idx=self.token_padding_idx, column_id_padding_idx=0
                )

        if incremental_state is not None:
            assert tuple(x_relation.shape) == (bsz, 1, x_length)
        else:
            assert tuple(x_relation.shape) == (bsz, x_length, x_length)

        return x_relation

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        table_states = self._get_input_buffer(incremental_state)
        if table_states is not None:
            table_states = [copy.deepcopy(table_states[i]) for i in new_order]
            incremental_state = self._set_input_buffer(incremental_state, table_states)
        return incremental_state

    def _get_input_buffer(self, incremental_state):
        return self.get_incremental_state(incremental_state, "decoder.table_states")

    def _set_input_buffer(self, incremental_state, buffer):
        return self.set_incremental_state(incremental_state, "decoder.table_states", buffer)


@register_model_architecture("bart_ours", "bart_ours_base")
def bart_ours_base_architecture(args):
    bart_base_architecture(args)


@register_model_architecture("bart_ours", "bart_ours_large")
def bart_ours_large_architecture(args):
    bart_large_architecture(args)
