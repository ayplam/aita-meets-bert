import pickle
from pathlib import Path

import pytest

from reddit_scrapr import Comment, Judgement


@pytest.fixture
def resource_path():
    return Path(__file__).parent.absolute() / "resources"


@pytest.fixture
def submission(resource_path):
    with open(resource_path / "submission.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def parsed_comments():
    return [
        Comment(body="A", score=2, judgement=Judgement.YTA),
        Comment(body="B", score=3, judgement=Judgement.YTA),
        Comment(body="C", score=10, judgement=Judgement.YTA),
        Comment(body="D", score=4, judgement=Judgement.NTA),
        Comment(body="E", score=6, judgement=Judgement.NAH),
        Comment(body="F", score=8, judgement=Judgement.ESH),
        Comment(body="F", score=1, judgement=Judgement.ESH),
    ]
