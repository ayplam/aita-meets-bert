from metaflow import FlowSpec, step, current, parallel_map, Parameter
import pandas as pd
from datetime import datetime
from reddit_scrapr import (
    get_date_ranges,
    pushshift_submission_query,
    comment_judgements,
    parse_submission,
    aita_score_summary,
    Post,
    Judgement
)

from aita_labels import (
    judgement_distribution,
    multiclass_label,
    multilabel_threshold_label,
    regression_label,
)

from typing import List

class AitaScrapeFlow(FlowSpec):
    """
    A flow to pull data from the AmITheAsshole subreddit and parse all the comments

    """
    
    data_dir = Parameter('data_dir',
                      help='Data directory to store and prevent repulls on subsequent runs',
                      default="./.reddit/top_level_comments")

    start_date = Parameter('start_date',
                      help='Date to start pulling data from, in YYYY-MM-DD',
                      default="2019-01-01")


    @step
    def start(self):
        """
        Estb

        """
        self.end_date = datetime.today().strftime("%Y-%m-%d")
        self.next(self.get_submissions)
        print(f"{current._flow_name} is starting. Grabbing data "
            "from {self.start_date} to {self.end_date")

    @step
    def get_submissions(self):
        date_ranges = get_date_ranges(self.start_date, self.end_date)
        self.posts: List[Post] = []
        
        for date_range in date_ranges:
            search_dict = search_dict = {
                "subreddit": "amitheasshole",
                "after": date_range.start,
                "before": date_range.end,
                "num_comments": ">100",
                "limit": 500,
                "sort_type": "num_comments",
                "sort": "desc"
            }
            
            query_data = pushshift_submission_query(search_dict)
            print(f"{len(query_data)} results for {date_range}")
            self.posts += [parse_submission(submission)
                for submission in query_data
            ]

        print(
            f"Found a total of {len(self.posts)} different posts between {self.start_date}-{self.end_date}"
        )
        self.next(self.get_comments)

    @step
    def get_comments(self):
        """Gets the top level comments in a post where users assess asshole-ness"""
        self.post_comments = parallel_map(
            lambda post: comment_judgements(post.id, data_dir=self.data_dir), self.posts,
        )

        self.next(self.aggregate_comments)

    @step
    def aggregate_comments(self):
        """Summarizes results of the comments of the post and creates labels"""
        aggregated_comments = parallel_map(
            lambda comments: aita_score_summary(comments), self.post_comments
        )
        
        assert len(aggregated_comments) == len(self.posts)
        post_aggcomments = [{**agg_comment, **post._asdict()}  for agg_comment, post in zip(aggregated_comments, self.posts)]
        
        self.asshole_results = pd.DataFrame(post_aggcomments).fillna(0)
        self.asshole_results["total"] = self.asshole_results[
            [x.name for x in Judgement]
        ].sum(axis=1)
        self.asshole_results["judgement"] = self.asshole_results[
            [x.name for x in Judgement]
        ].idxmax(axis=1)
        self.asshole_results["distribution"] = self.asshole_results.apply(
            lambda row: judgement_distribution(row), axis=1
        )

        self.asshole_results["label_multiclass"] = self.asshole_results[
            "distribution"
        ].map(lambda dd: multiclass_label(dd))
        self.asshole_results["label_multilabel"] = self.asshole_results[
            "distribution"
        ].map(lambda dd: multilabel_threshold_label(dd, 0.2))
        self.asshole_results["label_regression"] = self.asshole_results[
            "distribution"
        ].map(lambda dd: regression_label(dd))
        self.asshole_results["label_twoclass_multilabel"] = self.asshole_results[
            "label_multiclass"
        ].map(lambda x: [x[0] + x[-1], x[1] + x[-1]])
        

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print(f"{current.flow_name} is done.")


if __name__ == "__main__":
    AitaScrapeFlow()
