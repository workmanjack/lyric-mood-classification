<<<<<<< HEAD
# W266 Group Project: Lyric Mood Classification

UC Berkeley Masters of Information &amp; Data Science

W266 Natural Language Processing with Deep Learning Group Project

Team: Cyprian Gascoigne, Jack Workman, Yuchen Zhang

## Project Proposal

https://docs.google.com/document/d/1ofGlfFS2aUMOvsI7ZhUnTVFt6m2Gls2x_Sw_QMRh3vI/edit

Live Session Instructor: Daniel Cer
Group Members: Jack Workman, Yuchen Zhang, Cyprian Gascoigne

We plan to compare the accuracy of deep learning vs traditional machine learning approaches for classifying the mood of songs, using lyrics as features. We will use mood categories derived from Russell’s model of affect (from psychology, where mood is represented by vector in 2D valence-arousal space) and also calculate a valence-happiness rating. Mood categories will likely be happiness, anger, fear, sadness, and love (perhaps surprise, disgust).

We will test the extensibility of our deep learning model through genre classification, song quality prediction / album ratings, and additional text features such as part-of-speech tags, number of unique and repeated words and lines, lines ending with same words, etc.

Classifying mood (or “sentiment”) using textual features has been studied less than musical features. One reason may be in obtaining a large dataset legally (as lyrics are copyrighted material). The dataset we will use is the Million Song Dataset (MSD), a freely available million contemporary music track dataset. From MSD, we will use the Last.fm and musiXmatch datasets for song tags and lyrics. We will also use language detection to focus on English lyrics.

Algorithms we are considering in addition to RNN can be naive bayes, KNN, binary SVM, n-gram models like topK.

Previous work reached contradictory conclusion and employed smaller datasets and simpler methodology like td-idf weighted bag-of-words to derive unigrams, fuzzy clustering, and linear regression. Our proposed model approach (RNN) should have better accuracy.

Mood classification of lyrics can help in the creation of automatic playlists, a music search engine, labeling for digital music libraries, and other recommendation systems.

Paper References:
- Bandyopadhyay, Sivaji, Das, Dipankar, & Patra, Braja Gopal. (2015). Mood Classification of Hindi Songs based on Lyrics.
- Becker, Maria, Frank, Anette, Nastase, Vivi, Palmer, Alexis, and Staniek, Michael. (2017). Classifying Semantic Clause Types: Modeling Context and Genre Characteristics with Recurrent Neural Networks and Attention.
- Corona, Humberto & O’Mahony, Michael. (2015). An Exploration of Mood Classification in the Million Songs Dataset.
- Danforth, Christopher M. & Dodds, Peter Sheridan. (2009). Measuring the Happiness of Large-Scale Written Expression: Songs, Blogs, and Presidents
- Fell, Michael & Sporleder, Caroline. (2014). Lyrics-based Analysis and Classification of Music.
- Lee, Won-Sook & Yang, Dan. (2010). Music Emotion Identification from Lyrics
- Mihalcea, Rada & Strapparava, Carlo. (2012). Lyrics, Music, and Emotions.

## Dataset

- [Million Song Dataset (MSD)](https://labrosa.ee.columbia.edu/millionsong/)
- [Last.fm dataset](https://labrosa.ee.columbia.edu/millionsong/lastfm) (component of MSD) maps songs to user "tags" based on Last.fm user input. These tags can be anything from a human emotion to genre to animals. [Here is a list of unique tags]( https://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_unique_tags.txt).
- [MusiXmatch dataset](https://labrosa.ee.columbia.edu/millionsong/musixmatch) (component of MSD) is a collection of song lyrics mapped to song in a bag-of-words format

## Challenges

- MSD provides song->mood mapping at a song level (not per lyrical line). Is this okay?
- MSD (through musiXmatch) provides song->lyric mapping in a bag-of-words format. Preferably we would have lyrics provided as sequences for an LSTM. Can write a web crawler to extract these lyrics? Can we join the MSD with another dataset like https://github.com/walkerkq/musiclyrics/blob/master/billboard_lyrics_1964-2015.csv?

## Approach

- Create dataset mapping a song and its lyrics to one of the set of moods used in Corona's and O'Mahony's paper. Split dataset into train, dev, and test.
- Train an LSTM network with the train dataset and evaluate its performance on the dev dataset.
- Adjust parameters as needed. Evaluate performance on the test dataset.
- Sample the trained model to produce lyrics of a specified mood.

## Python Environment Setup

First, make sure you have Python 3.6+ installed.

Then, run through the following commands:

- `python -m venv .venv_w266_project`
- Windows: `.venv_w266_project\Scripts\activate.bat`
- Linux: `source .venv_w266_project/Scripts/activate`
- `pip install -r requirements.txt` - this will install all required packages and might take several minutes

## Downloading Data

The original dataset is quite large. Too large, in fact, to be stored in the github repo. To download the data, please run script download_data.py.

`python download_data.py`

This will download the data into the _data_ directory. This will take several minutes.

## Scraping Lyrics

We use the python package **lyricsgenius** for retrieving lyrics.

We attempt to match songs on all combinations of the MSD song title, MSD artist name, MXM song title, and MXM artist name. For more information, please see `scrape_lyrics.py`.

## Useful Links

(Python code for interacting with musicmatch lyrics)[https://github.com/tbertinmahieux/MSongsDB/tree/master/Tasks_Demos/Lyrics]

(Scraping song lyrics from Genius.com)[https://www.johnwmillr.com/scraping-genius-lyrics/}
