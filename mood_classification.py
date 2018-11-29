# project imports
from utils import read_file_contents, configure_logging, logger
from scrape_lyrics import LYRICS_TXT_DIR


# python and package imports
import pandas as pd
import numpy as np
import string
import os


# NLTK materials - make sure that you have stopwords and also punkt for some reason
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


LABELED_LYRICS_KEEP_COLS = ['msd_id', 'msd_artist', 'msd_title', 'is_english', 'lyrics_available',
                            'wordcount', 'lyrics_filename', 'mood', 'found_tags', 'matched_mood']


def prep_nltk():
    """
    These nltk corpuses are used by the lyrics_preprocessing function
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    return


def lyrics_preprocessing(lyrics):
    """
    Apply this function to any lyric file contents before reading for embeddings
    
    NLTK stopwords: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                     "you're", "you've", "you'll", "you'd", 'your', 'yours','yourself',
                     'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                     'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                     'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                     'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                     'do', 'does', 'did', 'doing','a', 'an', 'the', 'and', 'but', 'if',
                     'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                     'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                     'out','on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                     'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                     'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                     'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                     "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                     'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                     "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                     'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]
    
    Args:
        lyrics: str, some amount of lyrics
    """
    # https://stackoverflow.com/questions/17390326/getting-rid-of-stop-words-and-document-tokenization-using-nltk
    stop = stopwords.words('english') + list(string.punctuation)
    tokens = [i for i in word_tokenize(lyrics.lower()) if i not in stop]
    return tokens


def import_labeled_lyrics_data(csv_path, usecols=None):
    """
    Imports data from the provided path
    
    Assumes csv is a csv produced by label_lyrics.py
    
    Args:
        csv_path: str, path to csv to import data from
        usecols: list, cols to import (optional)
    """
    logger.info('Importing data from {0}'.format(csv_path))
    if not usecols:
        # we leave out the musixmatch id, artist, and title cols as well as the mood scoreboard cols
        # as they are unneeded for the cnn
        usecols = LABELED_LYRICS_KEEP_COLS
    df = pd.read_csv(csv_path, usecols=usecols)
    
    logger.info('imported data shape: {0}'.format(df.shape))
    
    return df

    
def filter_labeled_lyrics_data(df, drop=True):
    """
    Removes rows of data not applicable to this project's analysis

    Assumes df is from a csv produced by label_lyrics.py
    
    Args:
        df: pd.DataFrame
        drop: bool, flag to drop filtered cols or not to save memory
        
    Returns: filtered dataframe
    """
    logger.info('Data shape before filtering: {0}'.format(df.shape))
    
    df = df[df.is_english == 1]
    logger.info('Shape after is_english filter: {0}'.format(df.shape))

    df = df[df.lyrics_available == 1]
    logger.info('Shape after lyrics_available filter: {0}'.format(df.shape))
    
    df = df[df.matched_mood == 1]
    logger.info('Shape after matched_mood filter: {0}'.format(df.shape))

    if drop:
        # remove no longer needed columns to conserve memory
        df = df.drop(['is_english', 'lyrics_available', 'matched_mood'], axis=1)
        logger.info('Cols after drop: {0}'.format(df.columns))
        
    return df


def categorize_labeled_lyrics_data(df):
    """
    Creates a categorical data column for moods

    Thank you: https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers

    Assumes df is from a csv produced by label_lyrics.py
    
    Args:
        df: pd.DataFrame

    Returns: pd.DataFrame with categorical 'mood_cats' column
    """
    df.mood = pd.Categorical(df.mood)
    df['mood_cats'] = df.mood.cat.codes
    logger.info('Unique mood categories:\n{0}'.format(df['mood_cats'].unique()))
    logger.info('Shape after mood categorization: {0}'.format(df.shape))
    return df
    
    
def extract_lyrics(lyrics_filepath):
    """
    Extract lyrics from provided file path
    
    Args:
        lyrics_filepath: str, path to lyrics file
        
    Returns: str, lyrics or '' if path does not exist
    """
    # read in the lyrics of each song
    lyrics = ''
    if os.path.exists(lyrics_filepath):
        lyrics = read_file_contents(lyrics_filepath)[0]
    return lyrics


def extract_words(lyrics_series):
    """
    Concatenates all elements of lyrics_series and creates a list where
    each element is a single word.
    
    Args:
        words_series: pd.Series, series of song lyrics
        
    Returns: list
    """
    words = ' '.join(lyrics_series)
    return words

                 
def make_lyrics_txt_path(lyrics_filename, lyrics_dir=LYRICS_TXT_DIR):
    """
    The labeled_lyrics csv has the lyrics filename of each track
    without its extension or parent. This helper function links
    those missing pieces of info together.
    
    Args:
        lyrics_filename: str, name of lyrics file
        lyrics_dir: str, root dir of lyrics files
        
    Returns: str
    """
    return os.path.join(lyrics_dir, lyrics_filename) + '.txt'


def split_data(df):
    """
    Splits the supplied ndarray into three sections of
    """
    # thank you: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213
    # optional random dataframe shuffle
    #df = df.reindex(np.random.permutation(df.index))
    df_train, df_dev, df_test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    logger.info('df_train shape: {0}, pct: {1}'.format(df_train.shape, df_train.shape[0] / len(df)))
    logger.info('df_dev shape: {0}, pct: {1}'.format(df_dev.shape, df_dev.shape[0] / len(df)))
    logger.info('df_test shape: {0}, pct: {1}'.format(df_test.shape, df_test.shape[0] / len(df)))
    return df_train, df_dev, df_test


def compute_lyrics_cutoff(df):
    pctiles = df.wordcount.describe()
    logger.debug(pctiles)
    cutoff = int(pctiles[pctiles.index.str.startswith('75%')][0])
    logger.info('\nAll songs will be limited to {0} words'.format(cutoff))
    return cutoff


def normalize_lyrics(lyrics, max_length, lyrics_vectorizer):
    """
    Tokenize, process, shorten/lengthen, and vectorize lyrics
    """
    lyrics = lyrics_preprocessing(lyrics)
    if len(lyrics) > max_length:
        lyrics = lyrics[:max_length]
    else:
        lyrics += ['<PAD>'] * (int(max_length) - int(len(lyrics)))

    lyric_vector = lyrics_vectorizer.transform(lyrics)
    return lyric_vector


def split_x_y(df_train, df_dev, df_test):
    
    x_train = np.array(list(df_train.normalized_lyrics))
    y_train = pd.get_dummies(df_train.mood).values
    x_dev = np.array(list(df_dev.normalized_lyrics))
    y_dev = pd.get_dummies(df_dev.mood).values
    x_test = np.array(list(df_test.normalized_lyrics))
    y_test = pd.get_dummies(df_test.mood).values
  
    return x_train, y_train, x_dev, y_dev, x_test, y_test 


#### TODO!!!! Resolve circular references!!! We need dataset to spit out 
def build_labeled_lyrics_dataset(labeled_lyrics_csv, split=True):
    """
    Imports csv, filters unneeded data, and imports lyrics into a dataframe
    
    Assumes csv is from csv produced by label_lyrics.py

    Args:
        labeled_lyrics_csv: str, path to labeled_lyrics csv file
        split: bool, splits dataset into train, dev, and test if True; otherwise,
            returns complete DataFrame
        
    Returns: pd.DataFrame or (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    # import, filter, and categorize the data
    df = import_labeled_lyrics_data(labeled_lyrics_csv)
    df = filter_labeled_lyrics_data(df)
    df = categorize_labeled_lyrics_data(df)

    # import the lyrics
    # here we make use of panda's apply function to parallelize the IO operation
    df['lyrics'] = df.lyrics_filename.apply(lambda x: extract_lyrics(make_lyrics_txt_path(x)))
    logger.info('Data shape after lyrics addition: {0}'.format(df.shape))
    logger.info('Df head:\n{0}'.format(df.lyrics.head()))

    words = extract_words(df.lyrics)
    lyrics_vectorizer = lyrics2vec()
    lyrics_vectorizer.build_dataset(VOCAB_SIZE, words)
    # normalize the lyrics
    cutoff = compute_lyrics_cutoff(df)
    logger.info("Normalizing lyrics... (this will take a minute)")
    start = time.time()
    # here we make use of panda's apply function to parallelize the IO operation (again)
    df['normalized_lyrics'] = df.lyrics.apply(lambda x: normalize_lyrics(x, cutoff, lyrics_vectorizer))
    logger.info('data normalized ({0} minutes)'.format((time.time() - start) / 60))
    logger.debug(df.normalized_lyrics.head())

    # dumps some examples
    logger.info('\nExample of padding:')
    example = df.normalized_lyrics[df.normalized_lyrics.str.len() == cutoff].iloc[0]
    logger.info('\tFirst 5 tokens: {0}'.format(example[:5]))
    logger.info('\tLast 5 tokens: {0}.'.format(example[-5:]))

    logger.info('\nElapsed Time: {0} minutes'.format((time.time() - start) / 60))

    return df


def mood_classification():
    
    configure_logging(logname='mood_classification')
    prep_nltk()
    
    lyrics_vectorizer = lyrics2vec()
    # only extract words if we absolutely need to as it takes ~5 minutes
    datasets_loaded = False
    # first look for pickled datasets
    #datasets_loaded = lyrics_vectorizer.load_datasets()
    if not datasets_loaded:
        df = build_labeled_lyrics_dataset('data/labeled_lyrics_expanded.csv')
        return
        words = extract_words(df.lyrics)
        words = lyrics_vectorizer.extract_words(LYRICS_TXT_DIR, lyrics_preprocessing, words_file=VOCABULARY_FILE)
        lyrics_vectorizer.build_dataset(VOCAB_SIZE, words)
        lyrics_vectorizer.save_datasets()

    embeddings_loaded = False
    # only train embeddings if we absolutely need to as it takes a while!
    #embeddings_loaded = lyrics_vectorizer.load_embeddings()
    if not embeddings_loaded:
        lyrics_vectorizer.train(V=V)
        lyrics_vectorizer.save_embeddings()

    lyrics_vectorizer.plot_with_labels(os.path.join(LYRICS2VEC_DIR, 'embeddings_expanded.png'))


    configure_logging('lyrics_cnn')

    args = parse_args()

    np.random.seed(args.seed)
    if not args.skip_pickles and os.path.exists(LYRICS_CNN_DF_TRAIN_PICKLE):
        df_train = lyrics2vec.unpicklify(LYRICS_CNN_DF_TRAIN_PICKLE)
        df_dev = lyrics2vec.unpicklify(LYRICS_CNN_DF_DEV_PICKLE)
        df_test = lyrics2vec.unpicklify(LYRICS_CNN_DF_TEST_PICKLE)
    else:
        df = lyrics2vec.build_labeled_lyrics_dataset(args.labeled_lyrics_csv)
        df_train, df_dev, df_test = lyrics2vec.split_data(df)
        pickle_datasets(df_train, df_dev, df_test)
    
    x_train, y_train, x_dev, y_dev, x_test, y_test = split_x_y(df_train, df_dev, df_test)

    pretrained_embeddings = None
    if args.use_word2vec:
        logger.info('getting pretrained embeddings')
        pretrained_embeddings = get_pretrained_embeddings()
    
    cnn = LyricsCNN(
        # Data parameters
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=50000,
        # Model Hyperparameters
        embedding_size=300,
        filter_sizes=[3,4,5],
        num_filters=300,
        dropout=0.75,
        l2_reg_lambda=0.01,
        # Training parameters
        batch_size=64,
        num_epochs=12,
        evaluate_every=100,
        checkpoint_every=100,
        num_checkpoints=5,
        pretrained_embeddings=pretrained_embeddings,
        train_embeddings=False)
    
    model_summary_dir = os.path.join(cnn.output_dir, 'summaries')
    if os.path.exists(model_summary_dir):
        logger.info('Detected possible duplicate model: {0}'.format(cnn.output_dir))
        del_old_model = ''
        while True:
            print()
            del_old_model = input('You are about to overwrite old model data. Is this okay? (Y/N): ')
            done = del_old_model.lower() != 'y' or del_old_model.lower() != 'n'
            if del_old_model == 'n':
                logger.info('Okay, will not overwrite. Exiting...')
                return
            elif del_old_model == 'y':
                logger.info('Great! Moving on.')
                shutil.rmtree(model_summary_dir)
                break
            else:
                print('response not accepted')
    
    tb_proc = None
    if args.launch_tensorboard:
        logger.info('Launching tensorboard...')
        #best = ('w2v0', 'logs/tf/runs/Em-128_FS-3-4-5_NF-128_D-0.5_L2-0.01_B-64_Ep-20/summaries/')
        #best = ('w2v1_1', 'logs/tf/runs/Em-300_FS-3-4-5_NF-264_D-0.5_L2-0.01_B-128_Ep-10_W2V-1_V-50000/summaries/')
        #best = ('w2v1_2', 'logs/tf/runs/Em-300_FS-3-4-5_NF-300_D-0.5_L2-0.01_B-64_Ep-20_W2V-1_V-50000/summaries/')   # 52.74
        best = ('w2v1_3', 'logs/tf/runs/Em-300_FS-3-4-5_NF-300_D-0.75_L2-0.01_B-64_Ep-12_W2V-1_V-50000/summaries/')   # 54.30, 1.832
        # nope = ('w2v1_4', 'logs/tf/runs/Em-300_FS-3-4-5_NF-300_D-0.75_L2-0.1_B-64_Ep-12_W2V-1_V-50000/summaries')   # 47.36
        # nope = ('w2v1_5', 'logs/tf/runs/Em-300_FS-3-4-5_NF-300_D-0.75_L2-0.001_B-64_Ep-12_W2V-1_V-50000/summaries') # 54.55, 1.835 -- slightly more overtrained
        tb_cmd = build_tensorboard_cmd([best, ('new', model_summary_dir)])
        logger.info(tb_cmd)
        tb_proc = subprocess.Popen(tb_cmd.split())

    # Notes
    # * lower batch_size means less epochs; increase num_epochs inversely with batch_size to train for equal time
    # * higher num_filters means more memory_usage; lower batch_size to make up for it
    #     * num_filters = 512 is too much
        
    try:
        cnn.train(
            x_train,
            y_train,
            x_dev,
            y_dev,
            x_test,
            y_test)
    except Exception as e:
        logger.info('we had a problem...')
        logger.error(str(e))
    finally:
        if tb_proc:
            # kill tensorbroad process when user is ready
            user_input = ''
            while user_input.lower() != 'y':
                user_input = input("Ready to kill Tensorboard? (Y/N)")
                logger.info('user_input = {0}'.format(user_input))
            tb_proc.kill()
            logger.info('Killed tensorboard')

    logger.info('Model output dir: {0}'.format(model_summary_dir))
    logger.info('Done!')
                            
    return

    
    return
