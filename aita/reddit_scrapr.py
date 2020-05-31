import json
import pickle
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple
from urllib import parse, request

import pandas as pd
import praw
from praw.models import MoreComments
from praw.models.reddit.comment import Comment as PrawComment


def init_reddit(reddit_creds_file):
    try:
        with open(reddit_creds_file) as f:
            praw_args = json.load(f)

        return praw.Reddit(**praw_args)
    except FileNotFoundError:
        warnings.warn(
            f"Could not find {reddit_creds_file}, reddit methods will not work"
        )
        return None


# initialize the reddit API
reddit = init_reddit("reddit.creds")


class DateRange(NamedTuple):
    start: str
    end: str


class Post(NamedTuple):
    id: str
    title: str
    score: int
    selftext: str
    num_comments: int = None
    created_utc: int = None


class Judgement(Enum):
    YTA = 0
    NTA = 1
    NAH = 2
    ESH = 3


class Comment(NamedTuple):
    body: str
    score: int
    judgement: Judgement

    def __repr__(self):
        summary = self.body.replace("\n", " ")[:20]
        return f"<judgement={self.judgement}, score={self.score} : body={summary}...>"


def get_date_ranges(
    date_start: str, date_end: str, day_increment: int = 3
) -> List[DateRange]:
    """Creates a list of start/stop date ranges across a given period and increment

    The dates are intended to be inclusive at the start/end

    Parameters
    ----------
    date_start : str
        start date in format YYYY-MM-DD
    date_end : str
        end date in format YYYY-MM-DD
    day_increment : int, optional
        incremental frequency in days

    Returns
    -------
    List[DateRange]
        DateRanges across the provided period
    """

    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end) + pd.DateOffset(days=1)
    date_increments = list(
        pd.date_range(date_start, date_end, freq=f"{day_increment}d",)
    )

    if date_increments[-1] != date_end:
        date_increments.append(date_end)

    return [
        DateRange(
            dd.strftime("%Y-%m-%d"),
            (date_increments[idx + 1] + pd.DateOffset(days=-1)).strftime("%Y-%m-%d"),
        )
        for idx, dd in enumerate(date_increments[:-1])
    ]


def pushshift_submission_query(query_dict: Dict) -> Dict:
    """Calls pushshift API to obtain posts of interest

    Parameters
    ----------
    query_dict : Dict
        The query to pass to pushshift

    Returns
    -------
    Dict
        The returned query
    """
    sample_url = (
        "https://api.pushshift.io/reddit/submission/search/?"
        + f"{parse.urlencode(query_dict)}"
    )
    response = request.urlopen(sample_url)
    html = response.read()
    return json.loads(html)["data"]


def parse_submission(data):
    """Gets essential elements from a submission query by pushshift"""
    return Post(
        id=data["id"],
        title=data["title"],
        selftext=data.get("selftext"),
        score=data["score"],
        num_comments=data["num_comments"],
        created_utc=data["created_utc"],
    )


def is_comment_from_mod(comment_body: str) -> bool:
    if "I am a bot" in comment_body or "AUTOMOD" in comment_body:
        return True
    else:
        return False


def aita_comment_judgement(comment_body: str) -> str:
    """Get the judgement given a comment

    If more than one judgement was detected or words indicating the comment
    is from a bot, no judgement is returned

    Parameters
    ----------
    comment : str

    Returns
    -------
    str
        The judgement
    """
    # Ignore comments from bots
    if is_comment_from_mod(comment_body):
        return None

    final_judgement = [
        judgement.name for judgement in Judgement if judgement.name in comment_body
    ]
    if len(final_judgement) == 1:
        return final_judgement[0]
    else:
        return None


def parse_comment(comment: PrawComment) -> Comment:
    """Parses a comment for its body, score, and AITA judgement

    Parameters
    ----------
    comment : PrawComment
        The comment returned from praw

    Returns
    -------
    Comment

    """
    if is_comment_from_mod(comment.body):
        return Comment(body=comment.body, score=-9999, judgement=None)
    else:
        return Comment(
            body=comment.body,
            score=comment.score,
            judgement=aita_comment_judgement(comment.body),
        )


def comment_judgements(
    post_id: str, data_dir: str = "./.reddit/top_level_comments"
) -> List[Comment]:
    """Extracts top-level comments given a post_id with the reddit API

    Since extracting comments takes a significant amount of time,
    top_level_comments are saved to the local directory by ID. This
    unfortunately does create a TON of little files and could easily
    be improved with some batching

    Parameters
    ----------
    post_id : str
    data_dir : str, optional
        location to save data, by default "./.reddit/top_level_comments"

    Returns
    -------
    List[Comment]
        All top level comments in a post including their score and judgement
    """ """"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if the post as been pulled already
    comments_pkl = data_dir / f"{post_id}.pkl"
    if comments_pkl.exists():
        with open(comments_pkl, "rb") as f:
            comments = pickle.load(f)

        print(f"Loaded comment pickle for: {post_id}")
        return [Comment(**comment) for comment in comments]

    comments: List[Comment] = []

    for top_level_comment in reddit.submission(id=post_id).comments:
        if isinstance(top_level_comment, MoreComments):
            continue

        comments.append(parse_comment(top_level_comment))

    # Write to local to not pull the data again next time
    with open(comments_pkl, "wb") as f:
        # Store as dictionary to remove namedtuple dependency
        pickle.dump([comment._asdict() for comment in comments], f)

    return comments


def aita_score_summary(comments: List[Comment], min_score=25) -> Dict:
    """Calculates the overall score given a list of comments

    Scales the AITA judgement by the number of upvotes to calculate
    the total score for each judgement

    Parameters
    ----------
    comments : List[Comment]
        List of comments in the post

    Returns
    -------
    Dict
        A dictionary with the potential judgement as keys, ie:
        {"NTA": 210, "YTA": 34}
    """
    if not comments:
        return dict()
    else:
        df = pd.DataFrame([comment._asdict() for comment in comments])
        relevant_comments = df[
            df["judgement"].map(lambda x: x is not None) & (df["score"] > min_score)
        ]

        return relevant_comments.groupby("judgement").sum()["score"].to_dict()
