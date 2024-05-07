# SuperBowl49-Tweet-Analysis
A pipelie of complete analysis is performed over the tweets during th 49th Super Bowl with specific hashtags, including fan classification and tweet-related statistic prediction.

## Basic Statistics
A few basic analysis of the tweets were conducted. For example, the average number of tweets per hour for #gohawks is 296.7 spanning 570 hours, and that for #gopatriots is 53.3 spanning 441 hours. The difference between the tweet numbers of different fans indicates the potential imbalance inside the data for further analysis.

## Dataset
The source data can be found [here](https://ucla.app.box.com/s/24oxnhsoj6kpxhl6gyvuck25i3s4426d). The tweet file contains one tweet in each line and tweets are sorted with respect to their posting time. Each tweet is a JSON string that can be loaded in Python as a dictionary. For the following analysis, I only used data from the files `tweets_#gopatriots.txt` and `tweets_#gohawks.txt`.

### Resolve Imbalance
Given the fact that the tweets with #gohawks were many more than those with #gopatriots, we sample from `tweets_#gohawks.txt` to maintain the data balanced.

### Clean Data
The original tweet texts are messy and contain the hashtags that we might want to predict. We need a full clean for the text so that they can be further encoded into features. The basic cleaning is aimed to:
1. remove any hashtag and mention from the raw text, as those have too high correlation with the target
2. remove any html element like urls which cannot be well embedded or meaningless
3. remove the empty rows that have empty text or only spaces after previous steps, which may influence the model learning

## Feature Engineering
Two methods were selected for feature engineering: GLoVE and BERT. Considering the numtiliugual nature of tweet texts, we need to do necessary translation.

For GLoVE, as the embeddings were trained on the English corpus, the best solution is to use a translator(e.g., Romance-to-English translator) before feeding the features into the GLoVE encoder.  I tried the following code below:
```
romance_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ROMANCE-en")

def translate(translator, text):
    # return a sequence of translated text
    translations = translator(text, max_length=512)
    results = []
    for i in tqdm(range(len(translations))):
        translation = translations[i]
        results.append(translation['translation_text'])
    return results
```
However, the dataset is too large for a reasonable translation. Alternatively, I (automatically) exclude those non-English texts as they may not be included inside the GLoVE pre-trained embeddings.


For BERT, there is a pre-trained multi-lingual BERT mBERT available in huggingface, which can be used for encoding the text.

### GLoVE
GLoVE is a relatively weak encoder for text which is not very robust for messy data. To accomodate to this, I selected the tweets written in English and make lemmatization converion on the text for a deep clean. 

The strong cleaning by lemmatization guaranteed the effectiveness of further encoding, at the cost of introducing the potential bias as only English tweets are included in analysis. 

### m-BERT
The pre-trained model [m-BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased) has the capability to encode multi-lingual texts, which we can directly use for encoding the weakly cleaned tweets. The bias here comes from the pre-trained encoder itself.

## Classification 
Given the encoded tweets, I would like to train a model that can infer the fan class given the tweet posted by a user. The experiments included the following models:
- SVM (baseline)
- Naive logistic classifier
- LightGBM classifier
- Naive neural network classifier

Results showed that the Logistic classifier and lightGBM classifer both outperformed the baseline models on the GLoVE features.
A simple neural network can perform fairly well on the m-BERT encoded features, with which the other models could not work well. One explanation can be that those models could not resolve m-BERT features of high dimension.

## Prediction
Given the encoded tweets, I also want to predict the relevant statistics of a tweet, including retweets and likes. The experiments included the following models:
- L2-regularized regressor (baseline)
- LightGBM regressor
- Naive neural network regressor

Initially, LightGBM regressor outperformed the baseline model on both GLoVE and m-BERT features. 

However, the early-stage analysis showed that the retweets and likes both have a very extreme right-skewed distribution. Given the target values have a majority of 1's and high right-skew, we appled a log-transformation on the target to solve the rigght-skew. In particular, the transformation is $$y' = \log(y+1)$$
where the additional constant 1 is to avoid having $\log0$.
Thus, we make the reverse transformation when making a prediction $$y = \exp(y')-1$$

Results showed that the log-transformation greatly improved the performance of each model, showing the necessity of the appropriate feature engineering.

Besides, models working with the m-BERT encoded features showed better results, which is different from the classification task. This time, the neural network regressor obtained the best performances.

























