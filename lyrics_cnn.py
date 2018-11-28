"""
LyricsCNN class and supporting functions/variables
"""
# project imports
from lyrics2vec import LOGS_TF_DIR, read_file_contents, lyrics2vec
from scrape_lyrics import configure_logging, logger
from label_lyrics import CSV_LABELED_LYRICS
from index_lyrics import read_file_contents


# python and package imports
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os


# globals
LABELED_LYRICS_KEEP_COLS = ['msd_id', 'msd_artist', 'msd_title', 'is_english', 'lyrics_available',
                            'wordcount', 'lyrics_filename', 'mood', 'found_tags', 'matched_mood']

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
    logger.info('Shape after is_english filter:', df.shape)

    df = df[df.lyrics_available == 1]
    logger.info('Shape after lyrics_available filter:', df.shape)
    
    df = df[df.matched_mood == 1]
    logger.info('Shape after matched_mood filter:', df.shape)

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

                 
def make_lyrics_txt_path(lyrics_filename, lyrics_dir):
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


def split_data(data):
    """
    Splits the supplied ndarray into three sections of
    """
    # thank you: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213
    # optional random dataframe shuffle
    #df = df.reindex(np.random.permutation(df.index))
    return np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])


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
    lyrics = lyrics2vec.lyrics_preprocessing(lyrics)
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

    
def build_labeled_lyrics_dataset(labeled_lyrics_csv):
    """
    Imports csv, filters unneeded data, and imports lyrics into a dataframe
    
    Assumes csv is from csv produced by label_lyrics.py

    Args:
        labeled_lyrics_csv: str, path to labeled_lyrics csv file
        
    Returns: pd.DataFrame
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

    # split the data
    df_train, df_dev, df_test = split_data(df)
    logger.info('df_train shape: {0}, pct: {1}'.format(df_train.shape, df_train.shape[0] / len(df)))
    logger.info('df_dev shape: {0}, pct: {1}'.format(df_dev.shape, df_dev.shape[0] / len(df)))
    logger.info('df_test shape: {0}, pct: {1}'.format(df_test.shape, df_test.shape[0] / len(df)))

    # normalize the lyrics
    lyrics_vectorizer = lyrics2vec.InitFromLyrics()
    cutoff = compute_lyrics_cutoff(df)
    start = time.time()

    # here we make use of panda's apply function to parallelize the IO operation (again)
    df_train['normalized_lyrics'] = df_train.lyrics.apply(lambda x: normalize_lyrics(x, cutoff, lyrics_vectorizer))
    logger.info('train data normalized ({0} minutes)'.format((time.time() - start) / 60))
    logger.debug(df_train.normalized_lyrics.head())

    df_dev['normalized_lyrics'] = df_dev.lyrics.apply(lambda x: normalize_lyrics(x, cutoff, lyrics_vectorizer))
    logger.info('dev data normalized ({0} minutes)'.format((time.time() - start) / 60))
    logger.debug(df_dev.normalized_lyrics.head())

    df_test['normalized_lyrics'] = df_test.lyrics.apply(lambda x: normalize_lyrics(x, cutoff, lyrics_vectorizer))
    logger.info('test data normalized ({0} minutes)'.format((time.time() - start) / 60))
    logger.debug(df_test.normalized_lyrics.head())

    logger.info('\nExample of padding:')
    example = df_train.normalized_lyrics[df_train.normalized_lyrics.str.len() == cutoff].iloc[0]
    logger.info('\tFirst 5 tokens: {0}'.format(example[:5]))
    logger.info('\tLast 5 tokens: {0}.'.format(example[-5:]))

    logger.info('\nElapsed Time: {0} minutes'.format((time.time() - start) / 60))

    return df_train, df_dev, df_test


class LyricsCNN(object):
    """
    A CNN for mood classification of lyrics
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    
    Thank you to http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    and https://agarnitin86.github.io/blog/2016/12/23/text-classification-
    """
    def __init__(self, batch_size, num_epochs, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes,
                 num_filters, l2_reg_lambda=0.0, dropout=0.5, pretrained_embeddings=None, train_embeddings=False,
                use_timestamp=False, output_dir=None, evaluate_every=100, checkpoint_every=100, num_checkpoints=5):
        """
        Initializes class. Creates experiment_name. Initializes TF variables.
        
        Args:
            batch_size:
            num_epochs:
            sequence_length:
            num_classes: int, number of labels
            vocab_size: int
            embedding_size: int
            filter_sizes: list of ints
            num_filters: int
            l2_reg_lambda: float
            dropout: float, between 0 and 1
            pretrained_embeddings: ndarray, embeddings to use in model (optional: will train
                embeddings during execution if none are given)
            train_embeddings: bool, to train or not the embeddings during execution (optionaL:
                useful if you are supplying your own embeddings)
            use_timestamp: bool, True if you wish to identify this CNN with its timestamp at
                initialization rather than its parameter-unique name
            output_dir: str, root of dir to write files to
            evaluate_every: int, number of steps between dev loss & accuracy evaluation
            checkpoint_every: int, number of steps between each model checkpoint
            num_checkpoints: int, total number of checkpoints to save
                
        Returns: one LyricsCNN
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout = dropout
        self.pretrained_embeddings = pretrained_embeddings
        self.train_embeddings = train_embeddings
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.num_checkpoints = num_checkpoints
        
        self.experiment_name = self._build_experiment_name(timestamp=use_timestamp)
        self.output_dir = self._build_output_dir(output_dir)

        self.init_cnn()
        
        return
    
    def save_params(self, output):
        # dump params to json in case they need to be referenced later
        with open(os.path.join(out_dir, 'model_params.json'), 'w') as outfile:
            model_params = {
                'embedding_dim': embedding_dim,
                'filter_sizes': filter_sizes,
                'num_filters': num_filters,
                'dropout_keep_prob': dropout_keep_prob,
                'l2_reg_lambda': l2_reg_lambda,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'evaluate_every': evaluate_every,
                'checkpoint_every': checkpoint_every,
                'num_checkpoints': num_checkpoints,
                'train_embeddings': train_embeddings,
                'pretrained_embeddings': cnn.pretrained_embeddings
            }
            json.dump(model_params, outfile, sort_keys=True)

    def _build_output_dir(self, output_dir=None):
        """
        Builds the output directory. Makes parent directories if they do not
        exist.
        
        TODO: will fail if self.experiment_name is None.
        
        Returns: str, absolute output directory path
        """
        if not output_dir:
            output_dir = os.path.join(LOGS_TF_DIR, "runs")
        output_dir = os.path.abspath(os.path.join(output_dir, self.experiment_name))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _build_experiment_name(self, timestamp=False):
        """
        Constructs a parameter-unique name from model parameters
        
        What does "parameter-unique" mean? It means that your experiment name
        is guaranteed to be unique if no other models have been run with the
        same set of parameters.
        
        Note that the experiment name is used as an output directory so any model
        results with a matching parameter set will be overwritten.
        
        Args:
            timestamp: bool, if True, uses timestamp as unique key instead of
                parameter-unique name; if False, uses parameter-unique name
        
        Returns: str, experiment name
        """
        if timestamp:
            name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            name = 'Em-{0}_FS-{1}_NF-{2}_D-{3}_L2-{4}_B-{5}_Ep-{6}_W2V-{7}{8}_V-{9}'.format(
                self.embedding_size,
                '-'.join(map(str, self.filter_sizes)),
                self.num_filters,
                self.dropout,
                self.l2_reg_lambda,
                self.batch_size,
                self.num_epochs,
                1 if self.pretrained_embeddings else 0,
                '-Tr' if self.train_embeddings else '',
                self.vocab_size)

        return name
            
    def init_cnn(self):
        """
        Initializes all TF variables
        
        Returns: None
        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # control with l2_reg_lambda
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # for loading word2vec: https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            # optional: supply your own embeddings
            # (note: embeddings must match embedding_size!)
            if self.pretrained_embeddings:
                self.W = tf.get_variable(
                    shape=self.pretrained_embeddings.shape,
                    initializer=tf.constant_initializer(self.pretrained_embeddings),
                    trainable=self.train_embeddings,
                    name="W")
            else:
                self.W = tf.Variable(
                    initial_value=tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    trainable=self.train_embeddings,
                    name="W")
            
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, self.filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-{0}".format(self.filter_size)):
                # Convolution Layer
                filter_shape = [self.filter_size, self.embedding_size, 1, self.num_filters]
                Wconv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wconv")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    Wconv,
                    strides=[1, 1, 1, 1],
                    # options: 'SAME', 'VALID'
                    # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - self.filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            Wconv = tf.get_variable(
                "Wconv",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(Wconv)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, Wconv, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
        return

    def _batch_iter(self, data, shuffle=True):
        """
        Generates a batch iterator for the provided dataset
        
        Args:
            data: nparray, dataset
            shuffle: bool, shuffle data or not for each epoch
            
        Returns: batch iterator
        """
        data = np.array(data)
        data_size = len(data)
        self.num_batches_per_epoch = int((len(data) - 1) / self.batch_size) + 1
        print('num_batches_per_epoch = {0}'.format(self.num_batches_per_epoch))
        for epoch in range(self.num_epochs):
            print('***********************************************')
            print('Epoch {0}/{1}\n'.format(epoch, self.num_epochs))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(self.num_batches_per_epoch):
                print('-----------------------------------------------')
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, data_size)
                print('Epoch {0}/{1}, Batch {2}/{3} (start={4}, end={5})'.format(
                    epoch, self.num_epochs, batch_num, self.num_batches_per_epoch, start_index, end_index))
                yield shuffled_data[start_index:end_index]

    def _cnn_step(self, x_batch, y_batch, summary_op, train_op=None, summary_writer=None, step_writer=None):
        """
        A single step
        
        Args:
            x_batch: ndarray, inputs
            y_batch: ndarray, classes
            summary_op: tf summary op
            train_op: tf training operation (optional: if None, will not train model)
            summary_writer: tf.summary.FileWriter, to write tf summaries (optional)
            step_writer: csv.writer, to write step data (optional)
            
        Returns:
            time_str: str, time of step
            step: int, step number from tf
            loss: float, loss value of step from tf
            accuracy: float, acc value of step from tf
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.dropout
        }
        if train_op:
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, summary_op, self.loss, self.accuracy],
                feed_dict)
        else:
            step, summaries, loss, accuracy = sess.run(
                [global_step, summary_op, self.loss, self.accuracy],
                feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if summary_writer:
            summary_writer.add_summary(summaries, step)
        if step_writer:
            step_writer.writerow(['train' if dev_op else 'dev', time_str, step, loss, accuracy])
        return time_str, step, loss, accuracy

    def train(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        """
        Defines the TF graph for training the CNN
        
        Args:
            x_train: ndarray, training input
            y_train: ndarray, trainint labels
            x_dev: ndarray, validation dev input
            y_dev: ndarray, validation dev labels
            x_test: ndarray, validation test input
            y_test: ndarray, validation test labels
            
        Returns: None
        """
        print("Writing to {}\n".format(self.output_dir))
        
        with tf.Graph().as_default():
            
            session_conf = tf.ConfigProto()
            sess = tf.Session(config=session_conf)
            
            with sess.as_default():

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                summary_dir = os.path.join(self.output_dir, "summaries")
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, "train"), sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, "dev"), sess.graph)

                # Test summaries
                test_summary_op = tf.summary.merge([loss_summary, acc_summary])
                test_summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, "test"), sess.graph)

                # Step summaries
                csvfile = open(os.path.join(self.output_dir, 'step_data.csv'), 'w')
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['dataset', 'time', 'step', 'loss', 'acc'])

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(self.output_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Generate batches
                batches = self._batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
                # Training loop
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    # train for batch
                    self._cnn_step(x_batch, y_batch, summary_op=test_summary_op, 
                                   summary_writer=train_summary_writer, step_writer=csvwriter,
                                   train_op=train_op)
                    current_step = tf.train.global_step(sess, global_step)
                    # evaluate against dev
                    if current_step % evaluate_every == 0:
                        print("\nEvaluation:")
                        self._cnn_step(x_dev, y_dev, summary_op=dev_summary_op,
                                       summary_writer=dev_summary_writer, step_writer=csvwriter)
                        print()
                    # save model checkpoint
                    if current_step % checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                print("\nFinal Test Evaluation:")
                self._cnn_step(x_test, y_test, summary_op=test_summary_op,
                               summary_writer=test_summary_writer, step_writer=csvwriter)
                
        return


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--labeled-lyrics-csv', action='store', type=str, required=False, default=CSV_LABELED_LYRICS,
                        help='Path to labeled_lyrics csv to be used to build dataset')
    parser.add_argument('-s', '--seed', action='store', type=int, required=False, default=12,
                        help='Random seed to be used by numpy to ensure reproducibility')

    args = parser.parse_args()
    logger.info(args)    

    # simple arg sanity check
    if not os.path.exists(args.labeled_lyrics_csv):
        error = 'csv does not exist: {0}'.format(args.labeled_lyrics_csv)
        logger.error(error)
        raise Exception(error)

    return args
    
    
def main():
    
    configure_logging('lyrics_cnn')

    args = parse_args()

    np.random.seed(args.seed)
    df_train, df_dev, df_test = build_labeled_lyrics_dataset(args.labeled_lyrics_csv)
    x_train, y_train, x_dev, y_dev, x_test, y_test = split_x_y(df_train, df_dev, df_test)

    cnn = LyricsCNN(
        # Data parameters
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=50000,
        # Model Hyperparameters
        embedding_size=300,
        filter_sizes=[3,4,5],
        num_filters=128,
        dropout=0.5,
        l2_reg_lambda=0.01,
        # Training parameters
        batch_size=128,
        num_epochs=10,
        evaluate_every=100,
        checkpoint_every=100,
        num_checkpoints=5,
        pretrained_embeddings=None,
        train_embeddings=False)

    cnn.train(
        x_train,
        y_train,
        x_dev,
        y_dev,
        x_test,
        y_test)

    return