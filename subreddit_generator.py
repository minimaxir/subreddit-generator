#!/usr/bin/python3

import yaml
import pandas as pd
import re
import sys
from query import get_reddit_data
from textgenrnn import textgenrnn


def process_title_text(text):

    # Handle common HTML entities
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    return text


with open("config.yml", "r") as f:
    cfg = yaml.load(f)

reddit_data = get_reddit_data(cfg['project_id'],
                              cfg['subreddits'],
                              cfg['start_month'],
                              cfg['end_month'],
                              cfg['max_posts'])

reddit_data['title'] = reddit_data['title'].map(process_title_text)

# Write titles to a new file
with open('reddit_data.txt', 'w') as f:
    for post in reddit_data['title'].values:
        f.write(post + '\n')

if cfg['download_only']:
    sys.exit()

textgen = textgenrnn(name='reddit')
texts = reddit_data['title'].values
context_labels = reddit_data['context_label'].values

if cfg['new_model']:
    textgen.train_new_model(
        texts,
        context_labels=context_labels,
        num_epochs=cfg['num_epochs'],
        gen_epochs=cfg['gen_epochs'],
        batch_size=cfg['batch_size'],
        prop_keep=cfg['prop_keep'],
        rnn_layers=cfg['model_config']['rnn_layers'],
        rnn_size=cfg['model_config']['rnn_size'],
        rnn_bidirectional=cfg['model_config']['rnn_bidirectional'],
        max_length=cfg['model_config']['max_length'],
        dim_embeddings=cfg['model_config']['dim_embeddings'],
        word_level=cfg['model_config']['word_level'])
else:
    textgen.train_on_texts(
        texts,
        context_labels=context_labels,
        num_epochs=cfg['num_epochs'],
        gen_epochs=cfg['gen_epochs'],
        prop_keep=cfg['prop_keep'],
        batch_size=cfg['batch_size'])
