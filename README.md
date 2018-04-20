# Subreddit Generator

Train a neural network optimized for generating Reddit subreddit posts based off of any number of subreddits! Subreddit Generator automatically downloads as many Reddit submissions as you want from as many subreddits as you want. Blend wildly different subredditsfor hilarity!

Subreddit Generator is based off of [textgenrnn](https://github.com/minimaxir/textgenrnn), and trains the network using context labels for better submission synthesis.

## Usage

After installing the dependencies, open `config.yml` and at the top, add the `project_id` of a [Google Compute Engine](https://cloud.google.com/compute/) project owned by an account with access to [BigQuery](https://cloud.google.com/bigquery/) (free). Below that, specify the list of subreddit(s) you wish to retrieve. You can then change the time horizon of Reddit data to check using `start_month` and `end_month` (between December 2015 and December 2017), and change the number of top submissions retieved from each subreddit during that timeframe. You can also configure the `num_epochs` and whether to use the pretrained model or train a `new_model`. Then simply run:

```sh
python3 subreddit_generator.py
```

On the first time running the script, the console will ask you to authenticate with Google; do so.

The script will automatically save the weights (+ config and vocab for if `new_model`) for the trained model, which can then be loaded into textgenrnn and used anywhere.

The included `askreddit_weights.hdf5` file + relevant config info was trained on the top 50,000 /r/AskReddit submissions in 2017. You can load it and generate text from it simply with:

```python
from textgenrnn import textgenrnn
textgen=textgenrnn(weights_path="askreddit_weights.hdf5",
                   config_path="askreddit_config.json",
                   vocab_path="askreddit_vocab.json")
textgen.generate_samples()
```

You can view examples of the output at various temperatures in the `/example_output` folder.

## Warning

Google BigQuery gives 1 TB of data processing for free, and it will only charge for the data processed. For the default time range of 2017, BigQuery will consume  **8.68 GB** worth of data (regardless of how many subreddits and how many submissions you retrieve), which gives you plenty of leeway. Adjusting the time frame consumes data proportionately.

## Requirements

* textgenrnn
* tensorflow (either CPU or GPU flavors)
* pandas
* pandas-gbq

## Maintainer

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## License

MIT

## Credits

[Jason Baumgartner](https://twitter.com/jasonbaumgartne) for collecting the Reddit data, and [Felipe Hoffa](https://twitter.com/felipehoffa) for putting it in BigQuery.