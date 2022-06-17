"""
Copyright (c) <2022>, <Regina Nockerts>
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""

import pandas as pd
import numpy as np
import re
import regex
from numpy import random as rand
import emoji  # https://pypi.org/project/emoji/
from emosent import get_emoji_sentiment_rank
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.tokenize import TweetTokenizer  # Prefered: tokenizes a text, with extra controls
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# NOTE TO SELF: a lot of these functions should probably have been written for one tweet, then
# applied to the dataframe with .apply(): my_df['score_col'] = my_df['text_col'].apply(my_function)


### INDEX OF TOOLS ---------------------------------------

# NOTE: This function checks against the "Content" column, by choice.
# aa.term_check(term, df):

# NOTE: This function only flags positive instances; it may be run multiple times with different terms
# NOTE: This function starts with an input: to reset the index
# aa.flag_term(term, df, clean_col="ContentClean", flag_col="Flag", indx_warning=True, verby=False)

# NOTE: subset generator
# aa.subset_gen(df, n, seed=1080)

# NOTE: This function starts with an input: to reset the index
# aa.labeler(df, col="ContentClean", lab="ContentLabel", verby=False)

# NOTE: This function starts with an input: to reset the index
# aa.find_webs(df, text_col="Content", clean_col="ContentClean", web_col="https")

# NOTE: This function starts with an input: to reset the index
# aa.find_ats(df, clean_col="ContentClean", at_col="Mentions")

# NOTE: This function starts with an input: to reset the index
# NOTE: CapsRatio is ratio of capital letters to ContentClean.
# aa.count_caps(df, clean_col="ContentClean", n_caps_col="n_CapLetters", r_caps_col="CapsRatio", caps_col="AllCapWords")

# NOTE: This function starts with an input: to reset the index
# aa.find_rts (df="tweets_data", clean_col="ContentClean", rt_col="RT")

# NOTE: This function starts with an input: to reset the index
# aa.last_clean (df, text_col="ContentClean", indx_warning=True, verby=True)

# NOTE: This functions starts with input: box
# NOTE: This function returns THREE dataframes: superset only AND subset only AND inner/overlap, in that order
# a, b, c = aa.outer_df (superset, subset, silent="no")

# NOTE: Function which takes two lists and a reference indicating which class to calculate the TP, FP, and FN for.
# aa.classConfScores(y_true, y_pred, reference)

# NOTE: A function which takes the precision and recall of some model, and a value for beta, and returns the f_beta-score"""
# aa.fBetaScore(precision, recall, beta=1)

# NOTE: takes a string and goes through character by character to replace emoji with "!"
# aa.emojiToExcl (text)

# NOTE: Applies the emoji_cell function to a dataframe and returns a single list
# emoji_df(df, col="ContentClean")

# NOTE: gets the emosent score for an emoji, taking key errors into account
# emosent_score (emoj)

# NOTE: takes a text as string, breaks it into tokens and lemmatizes the tokens; returns list
# lemmatize_text(text)

# NOTE: takes a text as string and breaks it into bigrams and trigrams; finds the frequencies;
# NOTE: Returns two lists: bigram frequency, trigram frequency
# bi_tri_freq(ngram_text)



### CODE ---------------------------------------
# For help with regex codes, thanks to: https://regexr.com/4920fy

# NOTE: takes a text as string and breaks it into bigrams and trigrams; finds the frequencies;
# Returns two lists: bigram frequency, trigram frequency
def bi_tri_freq(ngram_text):
    # get lists of bi and trigrams [('word1', 'word2'), ('word2', 'word3'), ...]
    bigram_text = list(ngrams(ngram_text.split(), 2))
    bigram_dict = [' '.join(i) for i in bigram_text]
    trigram_text = list(ngrams(ngram_text.split(), 3))
    trigram_dict = [' '.join(i) for i in trigram_text]

    # Thanks to: https://stackoverflow.com/questions/49537474/wordcloud-of-bigram-using-python
    # Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(bigram_dict)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    vectorizer = CountVectorizer(ngram_range=(3, 3))
    bag_of_words = vectorizer.fit_transform(trigram_dict)
    vectorizer.vocabulary_
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    trigrams_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return bigrams_freq, trigrams_freq


# NOTE: takes a text as string, breaks it into tokens and lemmatizes the tokens; returns list
def lemmatize_text(text):
    text = text.lower()
    text = text.replace("'", "")
    w_tokenizer = nltk.tokenize.TweetTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# NOTE: gets the emosent score for an emoji, taking key errors into account
def emosent_score (emoj):
    try:
        a = get_emoji_sentiment_rank(emoj)
        return a['sentiment_score']
    except KeyError:
        return ""

# NOTE: Takes a text and returns a list of emojis, including compound emojis
def emoji_cell(text):  ## HELPER Function for emoji_df
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI['en'] for char in word):
            emoji_list.append(word)
    return emoji_list

# NOTE: Applies the emoji_cell function to a dataframe and returns a single list
def emoji_df(df, col="ContentClean"):
    a = df[col].apply(emoji_cell)
    a = sum(a, []) # With large lists, this might be less efficient than itertools
    a = list(dict.fromkeys(a))
    return a

# NOTE: takes a string and goes through character by character to replace emoji with "!"
def emojiToExcl (text):
    for i in text:
        if emoji.is_emoji(i):
            a = text.index(i)
            b = a+1
            text = text[:a] + "!" + text[b:]
    return text

# NOTE: Function which takes two lists and a reference indicating which class 
# to calculate the TP, FP, and FN for.
def classConfScores(y_true, y_pred, reference):
    Y_true = set([ i for (i,v) in enumerate(y_true) if v == reference])
    #print("Y_true:{}".format(Y_true))
    Y_pred = set([ i for (i,v) in enumerate(y_pred) if v == reference])
    #print("Y_pred:{}".format(Y_pred))
    TP = len(Y_true.intersection(Y_pred))
    # print(TP)
    FP = len(Y_pred - Y_true)
    FN = len(Y_true - Y_pred)
    return TP, FP, FN

# NOTE: A function which takes the precision and recall of some model, and a value for beta,
# and returns the f_beta-score"""
def fBetaScore(precision, recall, beta=1):
    return (1+beta**2) * precision * recall / (beta**2 * precision + recall)

# NOTE: This functions starts with input: box
# NOTE: This function returns THREE dataframes: superset only AND subset only AND inner/overlap, in that order
# a, b, c = outer_df (superset, subset)
def outer_df (superset, subset, silent="no"):
    silent = silent.lower()
    q = input("This function will match null values. To proceed, type: 'Y':")
    if q.lower() != "y":
        return "Error: null values."

    t_diff = subset.merge(superset, on="Content", how="outer", indicator=True)
    t_super = t_diff.loc[t_diff["_merge"] == "right_only"]
    t_super.reset_index(drop=True, inplace=True)
    t_sub = t_diff.loc[t_diff["_merge"] == "left_only"]
    t_sub.reset_index(drop=True, inplace=True)
    t_inner = t_diff.loc[t_diff["_merge"] == "both"]
    t_inner.reset_index(drop=True, inplace=True)
    if silent == "no":
        for i, tweet in enumerate(t_diff["Content"]):
            if i % 200 == 0:
                print("{}: {}".format(i, tweet))
    return t_super, t_sub, t_inner

# NOTE: This function starts with an input: to reset the index
def find_webs(df, text_col="Content", clean_col="ContentClean", web_col="https"):
    q = input("This function resets the index. To proceed, type: 'Y':")
    if q.lower() != "y":
        return "Error: Dataframe cannot be reindexed."
    else:
        df.reset_index(drop=True, inplace=True)

        # Finds all instances of "https://" in a tweet, grab from there to the next space, and put them in their own column
        for i, text in enumerate(df[text_col]):
            all_web = re.findall(r'https:\/\/\S+', text)
            a = ", ".join(all_web)
            df.loc[i, web_col] = a

            # Do it again; this time replace the the web address with a space and save the resulting text to its own column
            df.loc[i, clean_col] = re.sub(r'https:\/\/\S+', r' ', text)

            # Print out some progress checks
            if i % 10000 == 0:
                print("row count:", i)

# NOTE: This function starts with an input: to reset the index
def find_ats(df, clean_col="ContentClean", at_col="Mentions"):
    q = input("This function resets the index. To proceed, type: 'Y':")
    if q.lower() != "y":
        return "Error: Dataframe cannot be reindexed."
    else:
        df.reset_index(drop=True, inplace=True)

        # Find all instances of @mention in a de-webbed tweet, combine them into a string, and save them to their own column
        for i, text in enumerate(df[clean_col]):
            all_at = re.findall(r'\@\S+', text)
            all_at = ", ".join(all_at)
            df.loc[i, at_col] = all_at

            # Do it again, and this time replace the @mention with " " ["@m"] in the cleaned text - this can later be removed as a stopword
            df.loc[i, clean_col] = re.sub(r'\@\S+', r' ', text)

            # Print out some progress checks
            if i % 10000 == 0:
                print("replacing names:", i)

        # THIS would be nice...
        # for i, text in enumerate(df[clean_col]):
        #     # replace the "@m @m" with "@m" in the cleaned text - this can later be removed as a stopword
        #     df.loc[i, clean_col] = re.sub(r'[@m\s@m]+(?=[@m\s@m])', ' @m', text)
        #     # Example: text = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

        #     # Print out some progress checks
        #     if i % 10000 == 0:
        #         print("replacing @m:", i)

# NOTE: This function starts with an input: to reset the index
# NOTE: CapsRatio is ratio of capital letters to ContentClean.
def count_caps(df, clean_col="ContentClean", n_caps_col="n_CapLetters", r_caps_col="CapsRatio", caps_col="AllCapWords"):
    q = input("This function resets the index. To proceed, type: 'Y':")
    if q.lower() != "y":
        return "Error: Dataframe cannot be reindexed."
    else: 
        df.reset_index(drop=True, inplace=True)

        # Count the number of uppercase letters in each tweet, word by word, then...
        for i, tweet in enumerate(df[clean_col]):
            n = sum(word.isupper() for word in tweet)
            df.loc[i, n_caps_col] = int(n)
            # ... find the length of the tweet and ...
            m = len(tweet)
            # ... divide the number of uppercase by the length of the tweet to find the ratio
            df.loc[i, r_caps_col] = n/m

            # Split the tweet into words; if the word is all caps, add it to a list. Then...
            all_cap = []
            b = tweet.split(" ")
            for word in b:
                if word.isupper():
                    all_cap.append(word)
            # ... join the list into a string and save it to its own column.
            all_cap = ", ".join(all_cap)
            df.loc[i, caps_col] = all_cap

            # Print out a progress check
            if i % 10000 == 0:
                print("row count:", i)

# NOTE: This function starts with an input: to reset the index
def find_rts (df="tweets_data", clean_col="ContentClean", rt_col="RT"):
    q = input("This function resets the index. To proceed, type: 'Y':")
    if q.lower() != "y":
        return "Error: Dataframe cannot be reindexed."

    df.reset_index(drop=True, inplace=True)

    # Build a counter to keep track of the number of retweets
    counter = 0

    # Split the tweet into component words; check if the first word is "rt"; if it is, increment the counter and flag the row in the RT column
    for i, tweet in enumerate(df[clean_col]):
        a = tweet.lower().split(" ")
        if a[0] == "rt":
            counter += 1       
            df.loc[i,rt_col] = "rt"
    # Print the count of the number of "rt"s - this is unnecessary, but fast and nice as a progress check
    print("There are", counter, "stealth retweets in the data set.")

    # make a list of the index numbers of the rows flagged with "rt" in the RT column; drop those rows from the df; reset the index
    find_index = df[(df[rt_col] == "rt")].index
    df.drop(find_index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #This would probably be a more elegant solution:
    #df = df[df[rt_col] == 'rt'].reset_index(drop=True, inplace=True)

    #This should also work, would still need to reset the index:
    #df.drop(df.loc[df[rt_col] == "rt"], inplace=True)

    # reset the counter and recount the number of tweets that start with "rt" - should be 0
    # This is unnecessary, but fast and nice as a progress check
    counter = 0
    for i, tweet in enumerate(df[clean_col]):
        a = tweet.split(" ")
        if a[0] == "rt":
            counter += 1
    print("There are", counter, "stealth retweets REMAINING.")

# NOTE: This function starts with an input: to reset the index
def last_clean (df, text_col="ContentClean", indx_warning=True, verby=True):
    if indx_warning == True:
        q = input("This function resets the index. To proceed, type: 'Y':")
        if q.lower() != "y":
            return "Error: Dataframe cannot be reindexed."
    df.reset_index(drop=True, inplace=True)

    # Use regex to replace the default code "&amp;" with the "&" sign that BERT knows
    for i, tweet in enumerate(df[text_col]):
        # Replace '&amp;' with '&'
        df.loc[i, text_col] = re.sub(r'&amp;', '&', tweet)
        if verby == True:
            if i % 10000 == 0:
                print("& progress:", i)

    # Use regex to remove trailing spaces
    for i, tweet in enumerate(df[text_col]):
        # Remove trailing whitespace
        df.loc[i, text_col] = re.sub(r'\s+', ' ', tweet).strip()
        if verby == True:
            if i % 10000 == 0:
                print("trailing space progress:", i)


# NOTE: This function only flags positive instances; it may be run multiple times with different terms
# NOTE: This function starts with an input: to reset the index
def flag_term(term, df, clean_col="ContentClean", flag_col="Flag", indx_warning=True, verby=False):
    term = term.lower()
    if indx_warning == True:
        q = input("This function resets the index. To proceed, type: 'Y':")
        if q.lower() != "y":
            return "Error: Dataframe cannot be reindexed."
    df.reset_index(drop=True, inplace=True)

    # use reg exp to look for the term anywhere in the tweet, 
    # returns either the location or None. 
    for i, tweet in enumerate(df[clean_col]):
        x = re.search(term, tweet.lower())
        # if it returned a location (eg. not None), mark the row as "yes" in the flag_col 
        if x != None:
            df.loc[i, flag_col] = "yes"
        else:
            continue
        if verby == True:
            if i % 10000 == 0:
                print("Row:", i)
    print(df[flag_col].value_counts())

# Takes a string, lowercases it, and checks if that word is in each lowercased tweet, tweet by tweet, and counts the yeses
# NOTE: This function checks against the "Content" column, by choice.
def term_check(term, df, text_col="ContentClean"):
    term = term.lower()
    counter = 0
    for i, tweet in enumerate(df[text_col]):
        a = tweet.lower().split(" ")
        if term in a:
            counter += 1
    return term, counter

# NOTE: This function starts with an input: to reset the index
def labeler(df, col="ContentClean", lab="ContentLabel", verby = False):
    print("To end the labler session, enter 'ESC'")
    reset = input("Is it ok to reset the index? Y or N:")
    if reset.lower() == "n": 
        return "Error: Reindexing not allowed by user."
    df.reset_index(drop=True, inplace=True)

    # How to insert the label column without NoneType error?
    # orig_p = df.columns.get_loc(col)
    # position = orig_p + 1
    # l = list(df.columns)
    # name = col+"_label"
    
    # For each tweet, show the tweet in an input box, allowing user to input a label
    for i, tweet in enumerate(df[col]):
        ques = tweet + "  (CURRENT LABEL: " + str(df.loc[i, lab]).upper() + ")"
        answ = input(ques)
        # Check the input for escape term; if so, exit
        if answ == "ESC": 
            return "User escape"
        # Check the input for no answer; if so, skip
        if answ == "": 
            continue
        # Otherwise, save the input to its own column
        df.loc[i,lab] = answ
        
        # Print the label - this might be excessive
        if verby == True:
            print(answ)

    return df[lab].value_counts()

def subset_gen(df, n, seed=1080):
    # allow user to set random seed for reproducibility
    rng = rand.default_rng(seed)
    # create an index list of random numbers between 0 and the length of the dataframe
    indx = rng.uniform(0, df.shape[0], n)
    # create a new dataframe with just the rows indicated in the index list
    df = df.iloc[indx]  
    # reset the index of the new dataframe
    df = df.reset_index(drop=True)
    # save the new dataframe to csv
    df.to_csv('temp_subset_gen.csv')
    # print for progress check
    print("a dataframe and temp_subset_gen.csv of length {} have been created".format(df.shape[0]))
    return df


# create the sentiment intensity dictionary object
sid = SentimentIntensityAnalyzer()  #NOTE: this NEEDS to stay outside of the functions. I will be modifying it.

# creates the sentiment intensity dictionary
def vader_sid(tweet):
    return sid.polarity_scores(tweet)

# gets the compound score
def vader_sent_compound(tweet):
    scores = sid.polarity_scores(tweet)
    return scores["compound"]

# gets the classification of the compund score using the authors' suggested cutoff points
def vader_pred(tweet, pos_cut = 0.05, neg_cut = -0.05):
    scores = sid.polarity_scores(tweet)
    comp = scores["compound"]
    if comp >= pos_cut:
        return 2
    elif comp <= neg_cut:
        return 0
    else:
        return 1