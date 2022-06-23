
# Install
aaInstall.ipynb
Self-made functions: aardvark.py

# Code
1. Data Collection
   1. dataCollection.ipynb (data files in : archiveData, data)

2. Data Cleaning and Exploration
   1. dataCleaningA.ipynb (data files in : archiveData, dataLabelSets, data)
      - initial data exploration and visualization
      - remove web addresses, @mentions, retweets, duplicate tweets
      - creates data columns that might be useful for machine learning
      - refine dataset: remove irrelevant tweets by keyword: not relevant content, news sources
      - setup for labeling
   2. dataExploreRows.ipynb (data files in : archiveData)
      - data exploration
   3. dataLabeling.ipynb (term dict.txt, data files in : dataLabelSets)
      - a text input system to display and label data
   4. dataCleaningB.ipynb (data files in : archiveData, data)
      - Form large sets, delete unneeded rows, convert labels to numeric.
   5. dataEmoji.ipynb (data files in : archiveData, data)
      - Understand how emojis are treated, replace emojis with text, create emoji dictionary.
   6. dataSplitBalance
      - creates baanced, undersampled, and under- oversampled datasets with train/val/test splits

3. Sentiment Analysis: Majority Class Prediction
   1. baselineModel.ipynb
      - Predicts the majority class

4. Sentiment Analysis: VADER
   1. vaderBase.ipynb (data files in : dataVader)
      - runs VADER out of the box on x_test["ContentClean"] after changing the emoji to code but *before* updating the lexicon. So VADER will just treat it like unrecognizable text.
   2. vaderPrep.ipynb
      - creates the emoji code dictionary, proves update to VADER lexicon works
      - changes the format of the emoji codes for balanced sets in a way that can be used by both VADER and BERT
   3. vaderMod.ipynb
      - updates the VADER lexicon with improved emoji lexicon
      - updates the VADER lexicon with custom lexical values for words associated with a specific class
      - updates the VADER lexicon with with both emojis and the custom word values
      - model selection based on f1 score

5. Sentiment Analysis: BERT
   1. nlpBert/bertUnbalance.ipynb (data files in : dataBert)
      - Understanding the bert model, training model on the unbalanced dataset
   2. nlpBert/bertUnder.ipynb
      - training model on the undersampled dataset
   3. nlpBert/bertUnOv.ipynb
      - training model on the over- and undersampled dataset
   4. nlpBert/loadingModel.ipynb
      - model selection based on f1 score
      - running the best model (trained on unblanced data) on the unlabeled datasest
      - Visualizations with the full dataset

## Large Data Files
Large data files can be found at: https://ln5.sync.com/dl/9d3531d80/cwxgbsvb-2xbswrt8-b2ji56vi-7h7k58ej
or by contacting: r.a.nockerts@uva.nl

# About the Project
## Goal
language understanding: create a model that has a better understanding of the domain/field or refugee resettlement.
  * word ambiguity
  * extended context understanding

-> Hyp: tuned model will fall into fewer traps.

## Dataset
labeled dataset  ~1200
unlabeled dataset  ~200,000
Real world data 
* Afghani refugees
* political crisis --> complex, overwhelmingly negative language --> difficulty of models to distinguish sentiment

## Analyser Choices 
 1. ML with transfer learning
    1. BERT-base
    2. Others(ex: GPT3) - not used: future work
 2. Lexical model, no learning
    1. Vader
    2. Others (ex: TextBLob) - not used
 3. Combination
    1. TL embeddings into ML model - if time (unlikely) --> future work

# Future work
1. Do it again on another dataset to see if the proposition transfers across similarly structured topics. Teach the system to identify the primary aspect and label on that: 
   1. (sec)bad (sec)bad (sec)bad (primary)good --> positive
2. Try the BERT embeddings in machine learning models.
3. Comparison with stance analysis of the same data.

