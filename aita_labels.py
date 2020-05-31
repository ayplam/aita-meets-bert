import operator

from reddit_scrapr import Judgement

def key_of_max_value_in_dict(dd):
    return max(dd.items(), key=operator.itemgetter(1))[0]


def judgement_distribution(row):
    if row["total"] > 0:
        return {x: row[x.name] / row["total"] for x in Judgement}
    else:
        return {x: 0 for x in Judgement}


def multiclass_label(dd):
    highest_voted_judgement = key_of_max_value_in_dict(dd)
    return [1 if jj == highest_voted_judgement else 0 for jj in Judgement]


def multilabel_threshold_label(dd, ths):
    return [1 if dd[jj] >= ths else 0 for jj in Judgement]


def regression_label(dd):
    return [dd[jj] for jj in Judgement]