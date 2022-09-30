import json
import os
import sys

from constants import TEAM_INFO_KEYS, PLAYER_INFO_KEYS
from table_utils import parse_table_to_text


def split_filtered_relations(relations):
    team_relations = set()
    player_relations = set()
    for _, num, rel, label in relations:
        if isinstance(label, bool):
            team_relations.add((num[3], rel, label))
        elif isinstance(label, str):
            player_relations.add((num[3], rel, label))
        else:
            assert label is None
    return list(team_relations), list(player_relations)


def get_filtered_team_table(original, team_relations):
    home_relations = dict()
    vis_relations = dict()

    for num, rel, label in team_relations:
        if label:
            home_relations[rel] = num
        else:
            vis_relations[rel] = num

    keyset = set()
    keyset.update(home_relations.keys())
    keyset.update(vis_relations.keys())

    keys = []
    for key in TEAM_INFO_KEYS:
        if key in keyset:
            keys.append(key)

    data = [["", ], [original["home_line"]["TEAM-NAME"], ], [original['vis_line']['TEAM-NAME'], ]]
    for k in keys:
        data[0].append(TEAM_INFO_KEYS[k])
        data[1].append(home_relations.get(k, ""))
        data[2].append(vis_relations.get(k, ""))

    if len(vis_relations) == 0:
        data.pop(2)
    if len(home_relations) == 0:
        data.pop(1)
    if len(data) == 1:
        data = []
    return data


def get_filtered_player_table(original, player_relations):
    player_ids = set([label for num, rel, label in player_relations])
    player_ids = [str(x) for x in sorted([int(x) for x in player_ids])]

    per_player_relations = dict()
    for player_id in player_ids:
        per_player_relations[player_id] = {"PLAYER_NAME": original['box_score']['PLAYER_NAME'][player_id], }
    for num, rel, label in player_relations:
        assert rel.startswith("PLAYER-")
        rel = rel[len("PLAYER-"):]
        per_player_relations[label][rel] = num

    keyset = set()
    for relations in per_player_relations.values():
        keyset.update(relations.keys())

    keys = []
    for key in PLAYER_INFO_KEYS:
        if key in keyset:
            keys.append(key)

    return [[PLAYER_INFO_KEYS[k] for k in keys], ] + \
           [[per_player_relations[i].get(k, '') for k in keys] for i in player_ids]


if __name__ == "__main__":
    _, inp_dir, oup_dir = sys.argv

    for split in ['train', 'valid', 'test', ]:
        with open(os.path.join(inp_dir, f'{split}.json')) as f:
            original = json.load(f)
        with open(os.path.join(inp_dir, f'{split}.relations.json')) as f:
            relations = json.load(f)

        with open(os.path.join(oup_dir, f'{split}.data'), 'w') as f:
            for o, r in zip(original, relations):
                tr, pr = split_filtered_relations(r)
                team_table = get_filtered_team_table(o, tr)
                player_table = get_filtered_player_table(o, pr)
                text = "Team:\n{}\nPlayer:\n{}".format(
                    parse_table_to_text(team_table),
                    parse_table_to_text(player_table)
                )
                text = '\n'.join([line.strip() for line in text.splitlines() if len(line.strip()) > 0])
                f.write(text.replace("\n", " <NEWLINE> ").strip() + "\n")
