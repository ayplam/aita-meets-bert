import pickle
from pathlib import Path

import pytest

from aita.reddit_scrapr import Comment


@pytest.fixture
def resource_path():
    return Path(__file__).parent.absolute() / "resources"


@pytest.fixture
def submission_comments(resource_path):
    with open(resource_path / "submission_comments.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def parsed_comments():
    return [
        Comment(body="A", score=2, judgement="YTA"),
        Comment(body="B", score=3, judgement="YTA"),
        Comment(body="C", score=10, judgement="YTA"),
        Comment(body="D", score=4, judgement="NTA"),
        Comment(body="E", score=6, judgement="NAH"),
        Comment(body="F", score=8, judgement="ESH"),
        Comment(body="F", score=1, judgement="ESH"),
    ]
