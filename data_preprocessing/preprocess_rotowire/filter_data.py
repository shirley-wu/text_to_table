'''Improved preprocessing code based on https://github.com/harvardnlp/data2text/blob/master/data_utils.py'''
import argparse
import codecs
import json
import os
import pdb

from nltk import sent_tokenize

from text2num import text2num, NumberException

prons = set(["he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"])  # leave "it"
singular_prons = set(["he", "He", "him", "Him", "his", "His"])
plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])
too_frequent_number_words = {
    "a": 1,
    "an": 1,
    "a pair of": 2,
}

DELIM = "|"
HOME = "HOME"
AWAY = "AWAY"


def nword_condition_fn(next_n_word, condition_func, use_pdb=False, use_print=False):
    def f(tokens, i):
        ret = i + next_n_word - 1 < len(tokens) and condition_func(' '.join(tokens[i: i + next_n_word]))
        if ret:
            if use_pdb:
                pdb.set_trace()
            if use_print:
                print(' '.join(tokens))
        return ret

    return f
    # return lambda tokens, i: i + next_n_word - 1 < len(tokens) and condition_func(' '.join(tokens[i: i + next_n_word]))


TEMPLATE_1 = [
    # applicable for both teams and players
    (nword_condition_fn(1, lambda x: x.startswith("percent")), [
        "TEAM-FG3_PCT", "TEAM-FG_PCT", "FG3_PCT", "FG_PCT", "FT_PCT", ]),
    (nword_condition_fn(1, lambda x: x.startswith("point")), [
        "TEAM-PTS", "TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4", "PTS"]),
    (nword_condition_fn(1, lambda x: x.startswith("assist")), ["TEAM-AST", "AST", ]),
    (nword_condition_fn(1, lambda x: x.startswith("rebound") or x.startswith("board")), ["TEAM-REB", "REB", ]),
    (nword_condition_fn(1, lambda x: x.startswith("turnover")), ["TEAM-TOV", "TO", ]),
    # applicable to teams only (because team in text)
    (nword_condition_fn(2, lambda x: x.startswith("team assist")), ["TEAM-AST", ]),
    (nword_condition_fn(2, lambda x: x.startswith("team rebound")), ["TEAM-REB", ]),
    (nword_condition_fn(2, lambda x: x.startswith("team turnover")), ["TEAM-TO", ]),
    # applicable to players only (teams don't have such attributes)
    (nword_condition_fn(1, lambda x: x.startswith("block")), ["BLK", ]),
    (nword_condition_fn(2, lambda x: x.startswith("three pointer")), ["FG3M", ]),
    (nword_condition_fn(3, lambda x: x.startswith("three - pointer")), ["FG3M", ]),
    (nword_condition_fn(2, lambda x: x.startswith("field goal")), ["FGM", ]),
    (nword_condition_fn(2, lambda x: x.startswith("free throw")), ["FTM", ]),
    (nword_condition_fn(1, lambda x: x.startswith("minute")), ["MIN", ]),
    (nword_condition_fn(1, lambda x: x.startswith("foul")), ["PF", ]),
    (nword_condition_fn(1, lambda x: x.startswith("steal")), ["STL", ]),
]
TEMPLATE_2 = [
    # no conditions at all
    (lambda tokens, i: True,
     [("TEAM-WINS", "TEAM-LOSSES"), ("TEAM-PTS", "TEAM-PTS"),
      ("TEAM-PTS_QTR1", "TEAM-PTS_QTR1"), ("TEAM-PTS_QTR2", "TEAM-PTS_QTR2"),
      ("TEAM-PTS_QTR3", "TEAM-PTS_QTR3"), ("TEAM-PTS_QTR4", "TEAM-PTS_QTR4")]),
    # check if keyword exist in entire sentence
    (lambda tokens, i: "assist" in ' '.join(tokens), [("TEAM-AST", "TEAM-AST"), ]),
    (lambda tokens, i: "rebound" in ' '.join(tokens) or "board" in ' '.join(tokens), [("TEAM-REB", "TEAM-REB"), ]),
    (lambda tokens, i: "turnover" in ' '.join(tokens), [("TEAM-TOV", "TEAM-TOV"), ]),
    # check next word
    (nword_condition_fn(1, lambda x: x.lower().startswith("3pt")), [("FG3M", "FG3A"), ]),
    (nword_condition_fn(1, lambda x: x.lower().startswith("fg")), [("FGM", "FGA"), ]),
    (nword_condition_fn(1, lambda x: x.lower().startswith("ft")), [("FTM", "FTA"), ]),
    (nword_condition_fn(1, lambda x: x.startswith("shoot")), [("FGM", "FTA"), ]),
]
TEMPLATE_3 = [("FG3M", "FG3A"), ("FGM", "FGA"), ("FTM", "FTA"), ]  # no conditions at all


def get_ents(dat):
    players = set()
    teams = set()
    cities = set()
    for thing in dat:
        teams.add(thing["vis_name"])
        teams.add(thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["vis_city"] + " " + thing["vis_name"])
        teams.add(thing["vis_city"] + " " + thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["home_name"])
        teams.add(thing["home_line"]["TEAM-NAME"])
        teams.add(thing["home_city"] + " " + thing["home_name"])
        teams.add(thing["home_city"] + " " + thing["home_line"]["TEAM-NAME"])
        # special case for this
        if thing["vis_city"] == "Los Angeles":
            teams.add("LA" + thing["vis_name"])
        if thing["home_city"] == "Los Angeles":
            teams.add("LA" + thing["home_name"])
        # sometimes team_city is different
        cities.add(thing["home_city"])
        cities.add(thing["vis_city"])
        players.update(list(thing["box_score"]["PLAYER_NAME"].values()))
        cities.update(list(thing["box_score"]["TEAM_CITY"].values()))

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        entset.add(piece)

    all_ents = players | teams | cities

    return all_ents, players, teams, cities


def extract_entities(sent, all_ents, prons):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            sent_ents.append((i, i + 1, sent[i], True))  # is a pronoun
            i += 1
        elif sent[i] in all_ents:  # findest longest spans; only works if we put in words...
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            sent_ents.append((i, i + j - 1, " ".join(sent[i:i + j - 1]), False))
            i += j - 1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = set(["three point", "three - point", "three - pt", "three pt", "three - pointers", "three - pointer",
                   "three pointers"])
    return " ".join(sent[i:i + 3]) in ignores or " ".join(sent[i:i + 2]) in ignores


def extract_numbers(sent):
    sent_nums = []
    i = 0
    ignores = set(["three point", "three-point", "three-pt", "three pt"])
    # print sent
    while i < len(sent):
        toke = sent[i]
        a_number = False
        try:
            itoke = int(toke)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append((i, i + 1, int(toke), False))
            i += 1
        elif toke in number_words and not annoying_number_word(sent, i):  # get longest span  (this is kind of stupid)
            j = 1
            while i + j < len(sent) and sent[i + j] in number_words and not annoying_number_word(sent, i + j):
                j += 1
            try:
                sent_nums.append((i, i + j, text2num(" ".join(sent[i:i + j])), False))
            except NumberException:
                pass
            i += j
        elif toke in too_frequent_number_words:
            j = 1
            possible_num = toke
            while i + j < len(sent) and any(w.startswith(possible_num + ' ' + sent[i + j])
                                            for w in too_frequent_number_words):
                possible_num += ' ' + sent[i + j]
                j += 1
            if possible_num in too_frequent_number_words:
                sent_nums.append((i, i + j, too_frequent_number_words[possible_num], True))
            i += j
        else:
            i += 1
    return sent_nums


def match_numbers_to_templates(sent, sent_nums):
    # return: (nums w/o templates, nums with single word templates, nums with two word templates)
    matches = [False for _ in sent_nums]
    nums_with_single_word_templates = []
    nums_with_two_word_templates = []

    # Single word templates
    # Template 1: <NUMBER> <COLNAME> ; <NUMBER> - <COLNAME>
    for i, sent_num in enumerate(sent_nums):
        _, s, e, num, _ = sent_num
        colnames = []
        # check next word in NEXT_WORD_TO_RELATION
        if e < len(sent) and sent[e] != "-":
            for condition_func, keys in TEMPLATE_1:
                if condition_func(sent, e):
                    colnames += keys
        # check next next word in NEXT_WORD_TO_RELATION
        if e < len(sent) and sent[e].startswith("-"):
            for condition_func, keys in TEMPLATE_1:
                if condition_func(sent, e + 1):
                    colnames += keys
        # has at least one match
        if len(colnames) > 0:
            matches[i] = True
            nums_with_single_word_templates.append((sent_num, colnames))

    # Two word templates
    # This for loop is based on that `sent_nums` is always sorted
    for i in range(len(sent_nums) - 1):
        _, s0, e0, num0, _ = sent_nums[i]
        _, s1, e1, num1, _ = sent_nums[i + 1]
        colnames = []
        # Template 2: <NUMBER 1> - <NUMBER 2>
        if s1 == e0 + 1 and sent[e0] == "-":
            for condition_func, relations in TEMPLATE_2:
                if condition_func(sent, e1):
                    colnames += relations
        # Template 3: <NUMBER 1> - of/for - <NUMBER 2> ; <NUMBER 1> of/for <NUMBER 2>
        if (s1 == e0 + 3 and sent[e0] == "-" and sent[e0 + 2] == "-" and sent[e0 + 1] in ["of", "for", ]) or \
                (s1 == e0 + 1 and sent[e0] in ["of", "for", ]):
            colnames += TEMPLATE_3
        # has at least one match
        if len(colnames) > 0:
            matches[i] = True
            matches[i + 1] = True
            nums_with_two_word_templates.append((sent_nums[i], sent_nums[i + 1], colnames))

    nums_wo_templates = [(n, None) for n, m in zip(sent_nums, matches) if not m and not n[-1]]
    # if force match, cannot freely match: for a/a pair of/an
    return nums_wo_templates, nums_with_single_word_templates, nums_with_two_word_templates


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].items():
        if entname == v:
            keys.append(k)
    if len(keys) == 0:
        for k, v in bs["SECOND_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # take the earliest one
            keys.sort(key=lambda x: int(x))
            keys = keys[:1]
            # print "picking", bs["PLAYER_NAME"][keys[0]]
    if len(keys) == 0:
        for k, v in bs["FIRST_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # if we matched on first name and there are a bunch just forget about it
            return None
    # if len(keys) == 0:
    # print "Couldn't find", entname, "in", bs["PLAYER_NAME"].values()
    assert len(keys) <= 1, entname + " : " + str(list(bs["PLAYER_NAME"].values()))
    return keys[0] if len(keys) > 0 else None


def get_rels_one_word_template(entry, ents, nums, players, teams, cities):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    for i, ent in enumerate(ents):
        # if ent[3]:  # pronoun
        #     continue  # for now
        entname = ent[3]

        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname)
            if pidx is None:  # player might not actually be in the game or whatever
                continue

            for numtup, colnames in nums:
                strnum = str(numtup[3])
                if colnames is not None:
                    colnames = set(colnames).intersection(set(bs.keys()))
                else:
                    colnames = bs.keys()
                for colname in colnames:
                    col = bs[colname]
                    if col[pidx] == strnum:  # allow multiple for now
                        rels.append((ent, numtup, "PLAYER-" + colname, pidx))

        else:  # has to be city or team
            entpieces = entname.split()
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    is_home = False

            if is_home is None:
                continue
            linescore = entry["home_line"] if is_home else entry["vis_line"]

            for numtup, colnames in nums:
                strnum = str(numtup[3])
                if colnames is not None:
                    colnames = set(colnames).intersection(set(linescore.keys()))
                else:
                    colnames = linescore.keys()
                for colname in colnames:
                    val = linescore[colname]
                    if val == strnum:
                        rels.append((ent, numtup, colname, is_home))

    return rels


def get_rels_two_word_template(entry, ents, nums, players, teams, cities):
    if len(ents) == 0 or len(nums) == 0:
        return []

    rels = []
    bs = entry["box_score"]

    for i, ent in enumerate(ents):
        # if ent[3]:  # pronoun
        #     continue  # for now
        entname = ent[3]

        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname)
            if pidx is None:  # player might not actually be in the game or whatever
                continue
            for num1, num2, colnames in nums:
                strnum1 = str(num1[3])
                strnum2 = str(num2[3])
                for col1, col2 in colnames:
                    if col1 in bs and col2 in bs:
                        if bs[col1][pidx] == strnum1 and bs[col2][pidx] == strnum2:
                            rels.append((ent, num1, "PLAYER-" + col1, pidx))
                            rels.append((ent, num2, "PLAYER-" + col2, pidx))

        else:  # has to be city or team
            entpieces = entname.split()
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    is_home = False

            if is_home is None:
                continue
            linescore = entry["home_line"] if is_home else entry["vis_line"]
            linescore_other_team = entry["vis_line"] if is_home else entry["home_line"]

            for num1, num2, colnames in nums:
                strnum1 = str(num1[3])
                strnum2 = str(num2[3])
                for col1, col2 in colnames:
                    if col1 in linescore and col2 in linescore:
                        if col1 == col2:  # values for two different teams
                            if linescore[col1] == strnum1 and linescore_other_team[col1] == strnum2:
                                rels.append((ent, num1, col1, is_home))
                                rels.append((None, num2, col1, not is_home))
                            if linescore_other_team[col1] == strnum1 and linescore[col1] == strnum2:
                                rels.append((None, num1, col1, not is_home))
                                rels.append((ent, num2, col1, is_home))
                        else:
                            if linescore[col1] == strnum1 and linescore[col2] == strnum2:
                                rels.append((ent, num1, col1, is_home))
                                rels.append((ent, num2, col2, is_home))
    return rels


def resolve_pronouns(ents, prev_ents, players, teams, cities):
    prev_ents_players = []
    prev_ents_teams_cities = []

    if prev_ents is not None:
        for ent in prev_ents:  # (sent id, start, end, text, coref)
            ent_text = ent[3]
            ent_ref = ent if ent[-1] is None else ent[-1]
            if ent_text in players:
                prev_ents_players.append(ent_ref)
            if ent_text in teams or ent_text in cities:
                prev_ents_teams_cities.append(ent_ref)

    ret = []
    for ent in ents:  # (sent id, start, end, text, coref)
        if ent[-1]:
            pron = ent[3]
            if pron in singular_prons:
                ret += [(ent[0], ent[1], ent[2], ref[3], ref) for ref in prev_ents_players]
            if pron in plural_prons:
                ret += [(ent[0], ent[1], ent[2], ref[3], ref) for ref in prev_ents_teams_cities]
        else:
            ret.append(ent[:-1] + (None,))

    return ret


def prepend_sent_id_to_ents(idx, entities):
    return [(idx,) + x for x in entities]


def append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities):
    """
    appends tuples of form (sentence_tokens, [rels]) to candrels
    """
    candrels = []
    candents = []
    candnums = []
    sents = sent_tokenize(summ)
    prev_ents = None
    for j, sent in enumerate(sents):
        tokes = sent.split()

        ents = extract_entities(tokes, all_ents, prons)
        ents = prepend_sent_id_to_ents(j, ents)  # prepend sent id
        ents = resolve_pronouns(ents, prev_ents, players, teams, cities)  # remove pronoun, expand pronoun to coref
        candents += ents

        nums = extract_numbers(tokes)
        nums = prepend_sent_id_to_ents(j, nums)  # prepend sent id
        candnums += nums

        nums_wo_templates, nums_with_single_word_templates, nums_with_two_word_templates = \
            match_numbers_to_templates(tokes, nums)
        candrels += get_rels_one_word_template(entry, ents, nums_wo_templates, players, teams, cities)
        candrels += get_rels_one_word_template(entry, ents, nums_with_single_word_templates, players, teams, cities)
        candrels += get_rels_two_word_template(entry, ents, nums_with_two_word_templates, players, teams, cities)
        prev_ents = ents

    return candrels, candents, candnums


def get_datasets(path="../boxscore-data/preprocess_rotowire"):
    with codecs.open(os.path.join(path, "train.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)

    with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
        valdata = json.load(f)

    # with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
    with codecs.open(os.path.join(path, "test.json"), "r", "utf-8") as f:
        # original code use valid, but here we use test
        testdata = json.load(f)

    extracted_rels = []
    extracted_ents = []
    extracted_nums = []

    datasets = [trdata, valdata, testdata]
    for dataset in datasets:
        rels = []
        ents = []
        nums = []
        for i, entry in enumerate(dataset):
            summ = " ".join(entry['summary'])
            candrels, candents, candnums = append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities)
            rels.append(candrels)
            ents.append(candents)
            nums.append(candnums)

        extracted_rels.append(rels)
        extracted_ents.append(ents)
        extracted_nums.append(nums)

    return extracted_rels, extracted_ents, extracted_nums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    extracted_rels, extracted_ents, extracted_nums = get_datasets(args.path)
    for name, rels, ents, nums in zip(["train", "valid", "test", ], extracted_rels, extracted_ents, extracted_nums):
        for part, obj in zip(["relations", "entities", "numbers", ], [rels, ents, nums, ]):
            with open(os.path.join(args.path, "{}.{}.json".format(name, part)), 'w') as f:
                json.dump(obj, f)
