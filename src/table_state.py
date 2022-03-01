from dataclasses import dataclass
from typing import Optional, List, Union, Dict

import torch


@dataclass
class SpecialTokens:
    eos: Union[int, str]
    newline_token: Union[int, str]
    start_split_token: Union[int, str]
    space_split_token: Union[int, str]
    other_split_tokens: List[Union[int, str]]

    @property
    def all_split_tokens(self):
        return self.other_split_tokens + [self.start_split_token, self.space_split_token, ]


RELATIVE_COLUMN_STR_CHOICES = ["row_head", "col_head", ]


class TableState:
    def __init__(self, special_tokens: SpecialTokens, table_max_columns: Optional[int] = None,
                 return_relative_column_strs: Optional[List[str]] = None):
        # token info
        self.special_tokens = special_tokens
        assert isinstance(self.special_tokens, SpecialTokens)

        # state:
        #  [1] not in table (in_table: False; n_col: None; current_col: None)
        #  [2] counting the first line (in_table: True; n_col: int; current_col: None)
        #  [3] continueing the table (in_table: True; n_col: int; current_col: int)
        self.in_table = False
        self.n_col = None
        self.current_col = None

        # to calc absolute column id
        self.column_id_base = 0
        self.table_max_columns = table_max_columns
        assert self.table_max_columns is None or isinstance(self.table_max_columns, int), \
            "Wrong set {}".format(self.table_max_columns)

        # track history:
        self.token_history = []
        self.column_id_history = []
        self.row_id_history = []

        # to calculate relative column ids
        if return_relative_column_strs is None:
            return_relative_column_strs = RELATIVE_COLUMN_STR_CHOICES
        assert all([isinstance(x, str) for x in return_relative_column_strs])
        self.return_relative_column_strs = {x: i + 1 for i, x in enumerate(return_relative_column_strs)}

    def __repr__(self):
        return "{}(in_table={}, n_col={}, current_col={})".format(
            type(self).__name__, self.in_table, self.n_col, self.current_col
        )

    def step(self, token):
        if isinstance(token, torch.Tensor):
            token = token.item()

        # Step token
        if self.in_table:
            # NOTE: only use to space_split_token to count column
            if token == self.special_tokens.eos:
                self.exit_table()
            elif self.last_token == self.special_tokens.newline_token:  # just start a new line
                if token == self.special_tokens.start_split_token:  # still in the table
                    self.current_col = 0
                else:  # exit table
                    self.exit_table()
            elif self.current_col is None:  # counting the first line
                if token == self.special_tokens.space_split_token:
                    self.n_col += 1
            elif token == self.special_tokens.space_split_token:
                self.current_col += 1
        else:
            if self.last_token in [None, self.special_tokens.eos, self.special_tokens.newline_token, ]:
                # in decoding stage, the sentence will have preceding eos
                # in training stage, the sentence have no preceding eos, s.t. last_token is None
                if token == self.special_tokens.start_split_token:
                    self.enter_table()

        # Count column id
        # Team : <NEWLINE> | a | b | c | <NEWLINE> | 1 | 2 | 3 | <NEWLINE> Player : <NEWLINE> | d | e | f | <NEWLINE> | 4 | 5 | 6 |
        #   0  0     0     0 1 1 2 2 3 3     0     0 1 1 2 2 3 3     0        0   0     0     0 4 4 5 5 6 6     0     0 4 4 5 5 6 6
        if self.in_table and token not in [self.special_tokens.start_split_token,
                                           self.special_tokens.newline_token, self.special_tokens.eos, ]:
            current_col = self.n_col if self.current_col is None else self.current_col
            if token == self.special_tokens.space_split_token:  # already count to the next col, but still belong to last col
                current_col -= 1
            column_id = self.column_id_base + current_col + 1
        else:
            column_id = 0

        # Count row id
        if len(self.row_id_history) == 0:
            row_id = 0
        elif self.last_token == self.special_tokens.newline_token:
            row_id = self.row_id_history[-1] + 1
        else:
            row_id = self.row_id_history[-1]

        # Append history
        self.token_history.append(token)
        self.column_id_history.append(column_id)
        self.row_id_history.append(row_id)

    def enter_table(self):
        self.in_table = True
        self.n_col = 0
        self.current_col = None

    def exit_table(self):
        if self.current_col is not None:
            self.column_id_base += self.current_col
        self.in_table = False
        self.n_col = None
        self.current_col = None

    @property
    def last_token(self):
        if len(self.token_history) == 0:
            return None
        return self.token_history[-1]

    @property
    def ban_tokens(self):
        ret = []

        # ban newline & eos when:
        #  1. in state [2] / [3]; 2. last_token != space_split_token
        #  or 1. in state [3]; 2. current_col < n_col
        if self.in_table and self.last_token != self.special_tokens.space_split_token:
            ret += [self.special_tokens.newline_token, self.special_tokens.eos, ]
        if self.current_col is not None and self.current_col < self.n_col:
            ret += [self.special_tokens.newline_token, self.special_tokens.eos, ]

        if self.in_table:
            if self.last_token != self.special_tokens.newline_token:
                # ban all split token except space token when:
                #  1. in state [2] / [3]; 2. last_token != newline_token
                ret += self.special_tokens.other_split_tokens + [self.special_tokens.start_split_token, ]
            else:
                # ban all split token except start token when:
                #  1. in state [2] / [3]; 2. last_token == newline_token
                ret += self.special_tokens.other_split_tokens + [self.special_tokens.space_split_token, ]

        return sorted(set(ret))

    @property
    def enforce_tokens(self):
        ret = []

        # enforce newline when:
        #  1. in state [3]; 2. current_col == n_col; 3. last_token == space_split_token
        # or 1. in state [2]; 2. n_col == table_max_columns; 3. last_token == space_split_token
        if self.in_table:
            if self.current_col is not None:
                if self.current_col == self.n_col and self.last_token == self.special_tokens.space_split_token:
                    ret += [self.special_tokens.newline_token, self.special_tokens.eos, ]
            else:
                if self.table_max_columns is not None and self.n_col == self.table_max_columns and \
                        self.last_token == self.special_tokens.space_split_token:
                    ret += [self.special_tokens.newline_token, self.special_tokens.eos, ]

        return sorted(set(ret))

    @property
    def relative_column_states(self) -> Dict[str, List[int]]:
        current_row = self.row_id_history[-1]
        current_col = self.column_id_history[-1]
        row_head = [0, ] * len(self.token_history)
        col_head = [0, ] * len(self.token_history)
        if self.in_table and self.last_token != self.special_tokens.newline_token:
            next_col_id = current_col + (1 if self.last_token in self.special_tokens.all_split_tokens else 0)
            next_row_id = current_row
            for i, (col_id, row_id) in enumerate(zip(self.column_id_history, self.row_id_history)):
                if row_id == next_row_id:
                    break
                elif col_id == next_col_id:
                    if self.token_history[i] in self.special_tokens.all_split_tokens:
                        break
                    col_head[i] = 1
            for i, (col_id, row_id) in enumerate(zip(self.column_id_history, self.row_id_history)):
                if col_id > 0 and row_id == next_row_id:
                    if col_id == current_col:
                        break
                    if self.token_history[i] in self.special_tokens.all_split_tokens:
                        break
                    row_head[i] = 1

        return dict(
            row_head=row_head,
            col_head=col_head,
        )

    @property
    def relative_column_ids(self) -> torch.Tensor:
        ret = torch.zeros((len(self.token_history),), dtype=torch.long)

        relative_column_states = self.relative_column_states

        for k in RELATIVE_COLUMN_STR_CHOICES:
            if k in self.return_relative_column_strs:
                has_state = torch.tensor(relative_column_states[k], dtype=torch.bool)
                ret[has_state] = self.return_relative_column_strs[k]

        return ret


def get_relative_column_ids(tokens: torch.Tensor, special_tokens: SpecialTokens,
                            return_relative_column_strs: List[str], column_id_padding_idx=0) -> torch.Tensor:
    n_tokens = len(tokens)
    relative_column_ids = torch.full((n_tokens, n_tokens), column_id_padding_idx,
                                     dtype=tokens.dtype, device=tokens.device)
    state = TableState(special_tokens, return_relative_column_strs=return_relative_column_strs)
    for i, t in enumerate(tokens):
        state.step(t)
        relative_column_ids[i, :i + 1] = state.relative_column_ids
    return relative_column_ids


def get_batch_relative_column_ids(tokens: torch.Tensor, special_tokens: SpecialTokens,
                                  return_relative_column_strs: List[str], token_padding_idx=1,
                                  column_id_padding_idx=0):
    bsz, tokens_length = tokens.shape
    relative_column_ids = torch.full((bsz, tokens_length, tokens_length), column_id_padding_idx,
                                     dtype=tokens.dtype, device=tokens.device)
    for i in range(tokens.shape[0]):
        relative_column_ids_ = get_relative_column_ids(
            tokens[i, tokens[i] != token_padding_idx], special_tokens, return_relative_column_strs
        )

        n_padding = (tokens[i] == token_padding_idx).sum()
        if n_padding == 0:
            relative_column_ids[i] = relative_column_ids_
        elif tokens[i, 0] == token_padding_idx:  # left pad
            assert (tokens[i, n_padding:] != token_padding_idx).all()
            relative_column_ids[i, n_padding:, n_padding:] = relative_column_ids_
        else:  # right pad
            assert (tokens[i, :-n_padding] != token_padding_idx).all()
            relative_column_ids[i, :-n_padding, :-n_padding] = relative_column_ids_

    return relative_column_ids
