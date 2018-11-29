"""
LyricsCNN class and supporting functions/variables

These incredibly helpful sources were a big help when putting together the CNN:
* http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
* https://agarnitin86.github.io/blog/2016/12/23/text-classification-cnn

"""
# project imports
from scrape_lyrics import configure_logging, logger, LYRICS_TXT_DIR
from label_lyrics import CSV_LABELED_LYRICS
from index_lyrics import read_file_contents
from download_data import DATA_DIR
import lyrics2vec


# python and package imports
import tensorflow as tf
import pandas as pd
import numpy as np
import subprocess
import argparse
import datetime
import shutil
import time
import json
import csv
import os


# globals
LYRICS_CNN_DIR = os.path.join(lyrics2vec.LOGS_TF_DIR, 'lyrics_cnn')
LYRICS_CNN_DF_TRAIN_PICKLE = os.path.join(LYRICS_CNN_DIR, 'lyrics_cnn_df_train.pickle')
LYRICS_CNN_DF_DEV_PICKLE = os.path.join(LYRICS_CNN_DIR, 'lyrics_cnn_df_dev.pickle')
LYRICS_CNN_DF_TEST_PICKLE = os.path.join(LYRICS_CNN_DIR, 'lyrics_cnn_df_test.pickle')

# thank you: https://github.com/datasci-w266/2018-fall-main/blob/012607b576bb6b96182f819773f3b50155f31876/assignment/a3/lstm/rnnlm.py
# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


def pickle_datasets(df_train, df_dev, df_test):
    if not os.path.exists(LYRICS_CNN_DIR):
        os.makedir(LYRICS_CNN_DIR, exist_ok=True)
    lyrics2vec.picklify(df_train, LYRICS_CNN_DF_TRAIN_PICKLE)
    lyrics2vec.picklify(df_dev, LYRICS_CNN_DF_DEV_PICKLE)
    lyrics2vec.picklify(df_test, LYRICS_CNN_DF_TEST_PICKLE)
    return


def get_pretrained_embeddings():
    # get our pre-trained word2vec embeddings
    lyrics_vectorizer = lyrics2vec.lyrics2vec()
    embeddings_loaded = lyrics_vectorizer.load_embeddings()
    if embeddings_loaded:
        logger.info('embeddings shape: {0}'.format(lyrics_vectorizer.final_embeddings.shape))
    else:
        logger.info('failed to load embeddings!')
    return lyrics_vectorizer.final_embeddings


def build_tensorboard_cmd(experiments):
    """
    Constructs a tensorboard command out of <runs>
    Ex: !tensorboard --logdir 
        w2v0:logs/tf/runs/Em-128_FS-3-4-5_NF-128_D-0.5_L2-0.01_B-64_Ep-20/summaries/,
        w2v0-moodexp:logs/tf/runs/Em-300_FS-3-4-5_NF-64_D-0.5_L2-0.01_B-64_Ep-20_W2V-0-Tr_V-50000/summaries/

    Args:
        runs: list of tuples, each tuple is a run with (<name>, <path>)

    Returns: str, tensorboard logdir
    """
    logdir = ''
    for experiment in experiments:
        if len(experiment) != 2:
            logger.error('improperly formatted experiment: {0}'.format(experiment))
            continue
        name = experiment[0]
        path = experiment[1]
        logdir += '{0}:{1},'.format(name, path)
    # remove final comma
    logdir = logdir[:-1]
    return 'tensorboard --logdir {0}'.format(logdir)


class LyricsCNN(object):
    """
    A CNN for mood classification of lyrics
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    
    Thank you to http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    and https://agarnitin86.github.io/blog/2016/12/23/text-classification-
    """
    def __init__(self, batch_size, num_epochs, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes,
                 num_filters, l2_reg_lambda=0.0, dropout=0.5, pretrained_embeddings=None, train_embeddings=False,
                use_timestamp=False, output_dir=None, evaluate_every=100, checkpoint_every=100, num_checkpoints=5,
                graph=None):
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
            graph: tf graph, initialized and ready to go tf graph
                
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

        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        
        self.init_cnn()
        
        return
    
    def save_params(self, output):
        # dump params to json in case they need to be referenced later
        with open(output, 'w') as outfile:
            model_params = {
                'embedding_dim': self.embedding_size,
                'filter_sizes': self.filter_sizes,
                'num_filters': self.num_filters,
                'dropout_keep_prob': self.dropout,
                'l2_reg_lambda': self.l2_reg_lambda,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'evaluate_every': self.evaluate_every,
                'checkpoint_every': self.checkpoint_every,
                'num_checkpoints': self.num_checkpoints,
                'train_embeddings': self.train_embeddings,
                'pretrained_embeddings': self.pretrained_embeddings is not None,
                'evaluate_every': self.evaluate_every,
                'checkpoint_every': self.checkpoint_every,
                'num_checkpoints': self.num_checkpoints,
            }
            json.dump(model_params, outfile, sort_keys=True)
        return

    def _build_output_dir(self, output_dir=None):
        """
        Builds the output directory. Makes parent directories if they do not
        exist.
        
        TODO: will fail if self.experiment_name is None.
        
        Returns: str, absolute output directory path
        """
        if not output_dir:
            output_dir = os.path.join(lyrics2vec.LOGS_TF_DIR, "runs")
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
                1 if self.pretrained_embeddings is not None else 0,
                '-Tr' if self.train_embeddings else '',
                self.vocab_size)

        return name

    @with_self_graph
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
            if self.pretrained_embeddings is not None:
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
        logger.info('num_batches_per_epoch = {0}'.format(self.num_batches_per_epoch))
        for epoch in range(self.num_epochs):
            logger.info('***********************************************')
            logger.info('Epoch {0}/{1}\n'.format(epoch, self.num_epochs))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(self.num_batches_per_epoch):
                logger.info('-----------------------------------------------')
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, data_size)
                logger.info('Epoch {0}/{1}, Batch {2}/{3} (start={4}, end={5})'.format(
                    epoch, self.num_epochs, batch_num, self.num_batches_per_epoch, start_index, end_index))
                yield shuffled_data[start_index:end_index]

    def _cnn_step(self, sess, x_batch, y_batch, global_step, summary_op, train_op=None, summary_writer=None, step_writer=None):
        """
        A single step
        
        Args:
            sess: tf session, currently execution session
            x_batch: ndarray, inputs
            y_batch: ndarray, classes
            global_step: tf variable, stores the step count
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
            self.dropout_keep_prob: self.dropout if train_op else 1.0
        }
        if train_op:
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, summary_op, self.loss, self.accuracy],
                feed_dict)
        else:
            logger.info('Validation Step')
            step, summaries, loss, accuracy = sess.run(
                [global_step, summary_op, self.loss, self.accuracy],
                feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if summary_writer:
            summary_writer.add_summary(summaries, step)
        if step_writer:
            step_writer.writerow(['train' if train_op else 'dev', time_str, step, loss, accuracy])
        return time_str, step, loss, accuracy

    @with_self_graph
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
        logger.info("Writing to {}\n".format(self.output_dir))
        self.save_params(os.path.join(self.output_dir, 'model_params.json'))

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
                os.makedirs(checkpoint_dir, exist_ok=True)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = self._batch_iter(list(zip(x_train, y_train)))
            # Training loop
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # train for batch
                self._cnn_step(sess, x_batch, y_batch, global_step, summary_op=test_summary_op, 
                               summary_writer=train_summary_writer, step_writer=csvwriter,
                               train_op=train_op)
                current_step = tf.train.global_step(sess, global_step)
                # evaluate against dev
                if current_step % self.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    self._cnn_step(sess, x_dev, y_dev, global_step, summary_op=dev_summary_op,
                                   summary_writer=dev_summary_writer, step_writer=csvwriter)
                    logger.info('')
                # save model checkpoint
                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {}\n".format(path))

            logger.info("\nFinal Test Evaluation:")
            self._cnn_step(sess, x_test, y_test, global_step, summary_op=test_summary_op,
                           summary_writer=test_summary_writer, step_writer=csvwriter)
                
        return


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--labeled-lyrics-csv', action='store', type=str, required=False, default=CSV_LABELED_LYRICS,
                        help='Path to labeled_lyrics csv to be used to build dataset')
    parser.add_argument('-s', '--seed', action='store', type=int, required=False, default=12,
                        help='Random seed to be used by numpy to ensure reproducibility')
    parser.add_argument('--skip-pickles', action='store_true', required=False, default=False,
                        help='Do not use pickled datasets even if they\'re available')
    parser.add_argument('-t', '--launch-tensorboard', action='store_true', required=False, default=False,
                        help='Launch tensorboard on a subprocess to view this run')
    parser.add_argument('-w', '--use-word2vec', action='store_true', required=False, default=False,
                        help='Use pretrained word2vec embeddings')

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


if __name__ == '__main__':
    main()
