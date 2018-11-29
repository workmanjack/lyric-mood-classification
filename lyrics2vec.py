# project imports
from utils import read_file_contents, configure_logging, logger, picklify, unpicklify
from scrape_lyrics import LYRICS_TXT_DIR


# python and package imports
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import pickle
import string
import random
import time
import math
import os


# workaround for strange _tinker3 import error in matplotlib that didn't occur in jupyter notebook
# https://stackoverflow.com/questions/47778550/need-tkinter-on-my-python-3-6-installation-windows-10
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


UNKNOWN_TAG = 'UNK'
LOGS_TF_DIR = 'logs/tf'
LYRICS2VEC_DIR = os.path.join(LOGS_TF_DIR, 'lyrics2vec_expanded')
VOCABULARY_FILE = os.path.join(LYRICS2VEC_DIR, 'vocabulary.txt')
LYRICS2VEC_DATA_PICKLE = os.path.join(LYRICS2VEC_DIR, 'lyrics2vec_data.pickle')
LYRICS2VEC_COUNT_PICKLE = os.path.join(LYRICS2VEC_DIR, 'lyrics2vec_count.pickle')
LYRICS2VEC_DICT_PICKLE = os.path.join(LYRICS2VEC_DIR, 'lyrics2vec_dict.pickle')
LYRICS2VEC_REVDICT_PICKLE = os.path.join(LYRICS2VEC_DIR, 'lyrics2vec_revdict.pickle')
LYRICS2VEC_EMBEDDINGS_PICKLE = os.path.join(LYRICS2VEC_DIR, 'lyrics2vec_embeddings.pickle')


class lyrics2vec(object):
    """
    thank you: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    """
    REVERSED_DICTIONARY_PICKLE_NAME = 'revdict'
    DICTIONARY_PICKLE_NAME = 'dict'
    DATA_PICKLE_NAME = 'data'
    COUNT_PICKLE_NAME = 'count'
    EMBEDDINGS_PICKLE_NAME = 'embeddings' 

    def __init__(self, vocab_size, word_tokenizer_id):
        """
      word_tokenizer_id: int, id of word tokenizer (from mood_classification.lyrics_preprocessor)
              to be used when saving/restoring dataset
        """
        self.vocab_size = vocab_size
        self.word_tokenizer_id = word_tokenizer_id
        self.data_index = 0
        # dataset
        self.count = list()
        self.data = list()
        self.dictionary = dict()
        self.reversed_dictionary = dict()
    
    @classmethod
    def init_from_lyrics(cls, vocab_size, words, word_tokenizer_id, unpickle=True):
        """
        Initializer that builds dataset as well as inits
        
        Returns:
            lyrics2vec: initialized with dataset lyrics2vec
        """
        lyrics_vectorizer = lyrics2vec(vocab_size, word_tokenizer_id)
        loaded = False
        if unpickle:
            loaded = lyrics_vectorizer.load_dataset()
        if not loaded:
            lyrics_vectorizer.build_dataset(words)
            lyrics_vectorizer.save_dataset()
        return lyrics_vectorizer
  
    def build_dataset(self, words):
        """
        Process raw inputs into a dataset.
        
        Args:
          words: list of str, raw inputs
        
        Initializes the following class data members:
        * count: dict, maps each unique token to its int num of occurences in the dataset
        * dictionary: dict, maps each token to its int id
        * reversed_dictionary: dict, maps each int id to its token
        * data: list, int ids in order for all tokens in dataset
        
        Returns: None
        """
        logger.info('building lyrics2vec dataset')
        self.count = [[UNKNOWN_TAG, -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocab_size - 1))
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        unk_count = 0
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary[UNKNOWN_TAG]
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        return
    
    def transform(self, lyrics):
        """
        Maps each word in provided array to its integer ID from the constructed vocabulary
        
        Args:
          lyrics: list of strs
        
        Returns: list of ints
        """
        lyric_ids = list()
        for word in lyrics:
            lyric_ids.append(self.dictionary.get(word, 0))
            #print('{0} -> {1}'.format(word, lyric_ids[-1]))
        return lyric_ids

    def _build_lyrics2vec_dir(self):
        d = 'lyrics2vec_V-{0}_Wt-{1}'.format(self.vocab_size, self.word_tokenizer_id)
        d = os.path.join(LYRICS2VEC_DIR, d)
        os.makedirs(d, exist_ok=True)
        return d

    def _pickle_path(self, pickle_name):
        return os.path.join(self._build_lyrics2vec_dir(),
                            'lyrics2vec_{0}.pickle'.format(pickle_name))
    
    def load_dataset(self):
        loaded = False
        data_pickle = self._pickle_path(self.DATA_PICKLE_NAME)
        if os.path.exists(data_pickle):
            # we assume that if one exists then they all exist
            self.data = unpicklify(data_pickle)
            self.count = unpicklify(
                self._pickle_path(self.COUNT_PICKLE_NAME))
            self.dictionary = unpicklify(
                self._pickle_path(self.DICTIONARY_PICKLE_NAME))
            self.reversed_dictionary = unpicklify(
                self._pickle_path(self.REVERSED_DICTIONARY_PICKLE_NAME))
            loaded = True
            logger.info('lyrics2vec datasets successfully loaded via pickle')
        else:
            logger.warning('cannot load lyrics2vec datasets! no pickle data files found')
        return loaded
    
    def save_dataset(self):
        picklify(self.data, self._pickle_path(self.DATA_PICKLE_NAME))
        picklify(self.count, self._pickle_path(self.COUNT_PICKLE_NAME))
        picklify(self.dictionary, self._pickle_path(
            self.DICTIONARY_PICKLE_NAME))
        picklify(self.reversed_dictionary, self._pickle_path(
            self.REVERSED_DICTIONARY_PICKLE_NAME))
        logger.info('lyrics2vec datasets successfully pickled')
        return
    
    def save_embeddings(self):
        picklify(self.data, self._pickle_path(self.DATA_PICKLE_NAME))
        picklify(self.final_embeddings, self._pickle_path(self.EMBEDDINGS_PICKLE_NAME))
        return
    
    def load_embeddings(self):
        loaded = False
        path = self._pickle_path(self.EMBEDDINGS_PICKLE_NAME)
        if os.path.exists(path):
            self.final_embeddings = unpicklify(path)
            loaded = True
        else:
            logger.error('lyrics2vec cannot load final embeddings! no pickle data file found')
        return loaded
    
    def _generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # input word at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
                context[i * num_skips + j, 0] = buffer[target]  # these are the context words
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(data) - span) % len(data)
        return batch, context

    def train(self, batch_size=128, embedding_size=300, skip_window=4, num_skips=2, num_sampled=64):
        """
        Step 4: Build and train a skip-gram model.
        """
        #batch_size = 128
        #embedding_size = 128  # Dimension of the embedding vector.
        #skip_window = 1  # How many words to consider left and right.
        #skip_window = 4
        #num_skips = 2  # How many times to reuse an input to generate a label.
        #num_sampled = 64  # Number of negative examples to sample.
        
        logger.info('Building lyrics2vec graph')
        logger.info('vocab_size={0}, batch_size={1}, embedding_size={2}, skip_window={3},'
                    'num_skips={4}, num_sampled={4}'.format(self.vocab_size, batch_size,
                        embedding_size, skip_window, num_skips, num_sampled))
        
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent. These 3 variables are used only for
        # displaying model accuracy, they don't affect calculation.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        graph = tf.Graph()

        with graph.as_default():

            # Input data.
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocab_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=num_sampled,
                        num_classes=self.vocab_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                    valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

        logger.info('Beginning graph training')
        start = time.time()

        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            outdir = self._build_lyrics2vec_dir()
            writer = tf.summary.FileWriter(outdir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            logger.info('Initialized')

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = self._generate_batch(self.data, batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    logger.debug('Average loss at step {0}: {1}'.format(step, average_loss))
                    logger.debug('Time Elapsed: {0} minutes'.format((time.time() - start) / 60))
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = self.reversed_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to {0}:'.format(valid_word)
                        for k in range(top_k):
                            close_word = self.reversed_dictionary[nearest[k]]
                            log_str = '{0} {1},'.format(log_str, close_word)
                        logger.debug(log_str)

            self.final_embeddings = normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(os.path.join(outdir, 'metadata.tsv'), 'w') as f:
                for i in range(self.vocab_size):
                    f.write(self.reversed_dictionary[i] + '\n')

            # Save the model for checkpoints.
            saver.save(session, os.path.join(outdir, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = os.path.join(outdir, 'metadata.tsv')
            projector.visualize_embeddings(writer, config)

        writer.close()

        logger.info('Time Elapsed: {0} minutes'.format((time.time() - start) / 60))
        
        return

    def plot_with_labels(self):
        """
        Function to draw visualization of distance between embeddings.
        """
        
        logger.info('Beginning lyrics2vec label plotting')
        start = time.time()
        
        outfile = os.path.join(self._build_lyrics2vec_dir(), 'embeddings.png')
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(self.final_embeddings[:plot_only, :])
        labels = [self.reversed_dictionary[i] for i in range(plot_only)]

        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

            plt.savefig(outfile)
        
        logger.info('Elapsed Time: {0}'.format((time.time() - start) / 60))
        logger.info('saved plot at {0}'.format(outfile))

    def __repr__(self):
        return '<lyrics2vec()>'.format()


def main():

    prep_nltk()
    configure_logging(logname='lyrics2vec')
    
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
    
    return
    
    ### Batch Demo
    
    #batch_size = len(os.listdir(LYRICS_TXT_DIR))  # rough approximation of appropriate number of batches
    #batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size
    batch_size = 8
    print('batch_size =', batch_size)
    num_skips = 2  # How many times to reuse an input to generate a label.
    skip_window = 4  # take 4 before and 4 after for context window

    batch, labels = generate_batch(data, batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
    for i in range(batch_size):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
            reverse_dictionary[labels[i, 0]])
    print(data_index)
    print(batch)
    print(len(labels))
    
    return


if __name__ == '__main__':
    main()
    