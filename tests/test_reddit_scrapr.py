import pickle
from unittest import mock

import pytest

from aita.reddit_scrapr import (
    Judgement,
    aita_comment_judgement,
    aita_score_summary,
    comment_judgements,
    get_date_ranges,
    parse_comment,
)


@pytest.mark.parametrize(
    "start,stop,increment,expected_len",
    [("2020-01-01", "2020-01-10", 3, 4), ("2020-01-01", "2020-01-10", 2, 5)],
)
def test_date_ranges(start, stop, increment, expected_len):
    date_ranges = get_date_ranges(start, stop, increment)
    assert len(date_ranges) == expected_len
    assert date_ranges[0].start == start
    assert date_ranges[-1].end == stop


@pytest.mark.parametrize(
    "comment_type,expected_judgement,expected_score",
    [
        ("mod", None, -9999),
        ("multi", None, 237),
        ("nta", Judgement.NTA, 2282),
        ("unk", None, 467),
    ],
)
def test_parse_comment(resource_path, comment_type, expected_judgement, expected_score):
    with open(resource_path / f"comment_{comment_type}.pkl", "rb") as f:
        reddit_comment = pickle.load(f)

    comment = parse_comment(reddit_comment)
    assert comment.judgement == expected_judgement
    assert comment.score == expected_score


@pytest.mark.parametrize(
    "comment_body,expected_judgement",
    [
        ("YTA no questions asked", Judgement.YTA),
        ("NTA but it seems like YTA", None),
        ("Clearly a case of ESH", Judgement.ESH),
    ],
)
def test_aita_comment_judgement(comment_body, expected_judgement):
    assert aita_comment_judgement(comment_body) == expected_judgement


class MockReddit(object):
    """Empty class to mock reddit"""

    pass


@mock.patch("reddit_scrapr.reddit", side_effect=MockReddit())
def test_comment_judgements(mock_reddit, submission, tmpdir):
    mock_reddit.submission.return_value = submission
    comments = comment_judgements("fui7gp", data_dir=tmpdir)

    # Assert the mock was actually called
    assert mock_reddit.submission.call_count == 1

    assert len(comments) == 162

    # The first comment is the mod comment
    assert comments[0].judgement is None
    assert comments[0].score == -9999


def test_aita_score_summary(parsed_comments):
    score_summary = aita_score_summary(parsed_comments, min_score=0)
    assert len(score_summary) == 4
    assert score_summary["YTA"] == 15
    assert score_summary["ESH"] == 9

    score_summary = aita_score_summary(parsed_comments, min_score=5)
    assert len(score_summary) == 3
    assert score_summary["YTA"] == 10
    assert score_summary["NAH"] == 6
    assert score_summary["ESH"] == 8
