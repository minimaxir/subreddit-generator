import pandas as pd


def get_reddit_data(project_id, subreddits, start_month, end_month, max_posts):
    query = '''
    # standardSQL
    SELECT
    title,
    subreddit AS context_label
    FROM (
    SELECT
        title,
        subreddit,
        ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY score DESC)
        AS rank_num
    FROM
        `fh-bigquery.reddit_posts.*`
    WHERE
        _TABLE_SUFFIX BETWEEN "{}" AND "{}"
        AND LOWER(subreddit) IN ({})
        )
    WHERE
    rank_num <= {}
    '''

    query = query.format(start_month,
                         end_month,
                         str([x.lower() for x in subreddits])[1:-1],
                         max_posts)

    df = pd.read_gbq(query, project_id, dialect='standard')
    return df
