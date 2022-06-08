
# Install
aaInstall.ipynb
Self-made functions: aardvark.py

# Code
1. Data Collection
   1. dataCollection.ipynb (data files in : archiveData, data)

2. Data Cleaning and Exploration
   1. dataCleaningA.ipynb (data files in : archiveData, dataLabelSets, data)
   2. dataExploreRows.ipynb (data files in : archiveData)
   3. dataLabeling.ipynb (term dict.txt, data files in : dataLabelSets)
   4. dataCleaningB.ipynb (data files in : archiveData, data)

3. Sentiment Analysis: VADER
   1. vader.ipynb (data files in : dataVader)

4. Sentiment Analysis: BERT
   1. bertPreprocess.ipynb (data files in : datBert)
   2. bertExplore.ipynb (data files in : datBert) **
   3. bertYT.ipynb (data files in : datBert)


# Constaints are
* Sentiment analysis
* Tweets

# Prove sent.analyser
If Untuned --> bad results
If tuned --> better results

Q: Is untuned "good enough"?
  * definition of "good enough"?
Q: Is tuned "good enough"?

-> Hyp: tuned model will fall into fewer traps.

## Dataset
labeled dataset - 1200  (L)
unlabeled dataset - 300.000 (U)
Realworld data 
* Afghani refugees
* political crisis --> overwhelmingly negative language --> difficulty of models to distinguish sentiment
* language understanding: cerate a model that has a better understandiung of the field
  * word ambiguity
  * extended context understanding



--
# Analyser Choices 
 1. ML with transfer learning
    1. BERT
    2. Others(ex: GPT3)
 2. Lexical model, no learning
    1. Vader
    2. OThers (ex: TextBLob)
 3. Combination
    1. TL embeddings into ML model


# Development
1. For development Find/create a smaller dataset


# Future work
1. Do it again on another dataset to see if the proposition transfers across similarly structured topics. Teach the system to identify the primary aspect and label on that: 
   1. (sec)bad (sec)bad (sec)bad (primary)good --> positive


End result.

