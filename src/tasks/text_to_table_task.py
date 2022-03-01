import itertools
import logging
import os
from argparse import Namespace
from typing import List

import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.encoders import build_bpe
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from src.constrained_table_search import ConstrainedTableBeamSearch
from src.table_state import SpecialTokens, RELATIVE_COLUMN_STR_CHOICES, get_batch_relative_column_ids

logger = logging.getLogger(__name__)


@register_task("text_to_table_task")
class TextToDataTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

        # get special tokens
        start_split_token = args.split_token.strip()
        space_split_token = " " + start_split_token
        newline_token = args.newline_token

        # get bpe
        bpe = build_bpe(Namespace(bpe="gpt2"))
        # convert tokens
        start_split_token = tgt_dict.indices[bpe.encode(start_split_token)]
        space_split_token = tgt_dict.indices[bpe.encode(space_split_token)]
        newline_token = tgt_dict.indices[bpe.encode(newline_token)]
        # ban other split tokens in text
        other_split_tokens = [k.replace("Ġ", " ") for k in bpe.bpe.encoder if args.split_token.strip() in k]
        other_split_tokens.remove(" " + args.split_token.strip())
        other_split_tokens.remove(args.split_token.strip())
        other_split_tokens.remove("<|endoftext|>")  # handle special token
        other_split_tokens = [tgt_dict.indices[bpe.encode(k)] for k in other_split_tokens]
        # special tokens
        self.special_tokens = SpecialTokens(
            eos=self.target_dictionary.eos(),
            newline_token=newline_token,
            start_split_token=start_split_token,
            space_split_token=space_split_token,
            other_split_tokens=other_split_tokens,
        )

        self.table_max_columns = args.table_max_columns

    @classmethod
    def add_args(cls, parser):
        super(TextToDataTranslationTask, TextToDataTranslationTask).add_args(parser)
        parser.add_argument("--split-token", default="|")
        parser.add_argument("--newline-token", default="\n")
        parser.add_argument("--table-max-columns", type=int, required=True)
        parser.add_argument("--unconstrained-decoding", default=False, action="store_true")
        parser.add_argument("--return-relative-column-strs", default=None, nargs="+",
                            choices=RELATIVE_COLUMN_STR_CHOICES)

    def build_model(self, args):
        model = super().build_model(args)
        if hasattr(model, "set_special_tokens") and callable(model.set_special_tokens):
            model.set_special_tokens(self.special_tokens)
        return model

    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if self.args.unconstrained_decoding:
            return super().build_generator(
                models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
            )

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        assert self.table_max_columns is not None
        search_strategy = ConstrainedTableBeamSearch(
            self.target_dictionary, self.special_tokens, self.table_max_columns
        )

        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = SequenceGenerator
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        if not self.args.unconstrained_decoding:
            assert constraints is None
            bsz = sample['net_input']['src_tokens'].shape[0]
            constraints = torch.zeros((bsz, 0), dtype=torch.float)
        return super(TextToDataTranslationTask, self).inference_step(
            generator, models, sample, prefix_tokens=prefix_tokens, constraints=constraints,
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
            special_tokens=self.special_tokens,
            return_relative_column_strs=self.args.return_relative_column_strs,
        )


def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        special_tokens=None,
        return_relative_column_strs=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return TableWithColumnIdLanguagePairDataset(
        special_tokens,
        return_relative_column_strs,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


class TableWithColumnIdLanguagePairDataset(LanguagePairDataset):
    def __init__(self, special_tokens: SpecialTokens, return_relative_column_strs: List[str], *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self.special_tokens = special_tokens
        self.return_relative_column_strs = return_relative_column_strs

    def collater(self, samples, pad_to_length=None):
        res = super(TableWithColumnIdLanguagePairDataset, self).collater(samples, pad_to_length=pad_to_length)
        if len(samples) > 0:
            if self.return_relative_column_strs is not None:
                res['net_input']['prev_output_relative_column_ids'] = get_batch_relative_column_ids(
                    res['net_input']['prev_output_tokens'], self.special_tokens,
                    self.return_relative_column_strs, token_padding_idx=self.src_dict.pad(),
                )
        return res
