# project files
import index_lyrics
import label_lyrics
import lyrics2vec
import lyrics_cnn


# python packages
import tensorflow as tf
import pandas as pd
import numpy as np
import unittest
import shutil
import json
import csv
import os


class TestIndexLyrics(unittest.TestCase):

    test_txt = 'test.txt'
    input_csv = 'test_input.csv'
    output_csv = 'test_output.csv'

    def tearDown(self):
        if os.path.exists(self.test_txt):
            os.remove(self.test_txt)
        if os.path.exists(self.input_csv):
            os.remove(self.input_csv)
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def test_read_file_contents(self):
        # txt
        expected_contents = ('hello\nworld\n123', 'default')
        with open(self.test_txt, 'w') as f:
            f.write(expected_contents[0])
        actual_contents = index_lyrics.read_file_contents(self.test_txt)
        self.assertEqual(expected_contents, actual_contents)
        # json
        expected_contents = ({'hello': 'world', '1': {'2': '3', '4': '5'}}, 'default')
        with open(self.test_txt, 'w') as f:
            json.dump(expected_contents[0], f)
        actual_contents = index_lyrics.read_file_contents(self.test_txt, read_json=True)
        self.assertEqual(expected_contents, actual_contents)

    def test_add_col_if_dne(self):
        data = {'a': ['b', 'c'], 'd': ['e', 'f']}
        df = pd.DataFrame(data)
        expected_df = pd.DataFrame({'a': ['b', 'c'], 'd': ['e', 'f'], 'g': ['h', 'h']})
        actual_df = index_lyrics.add_col_if_dne(df, 'g', 'h')  # should be added
        actual_df = index_lyrics.add_col_if_dne(df, 'a', 'b')  # should not be added
        self.assertEqual(expected_df.to_dict(), actual_df.to_dict())

    def test_index_lyrics(self):
        with open(self.input_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['msd_artist', 'msd_title'])
            writer.writerow(['10cc', "I'm Not In Love"])
        # test 1: vanilla input csv with no additional columns
        index_lyrics.index_lyrics(self.input_csv, self.output_csv)
        expected_rows = [
            ['msd_artist', 'msd_title', 'is_english', 'lyrics_available', 'wordcount', 'lyrics_filename'],
            ['10cc', "I'm Not In Love", '1', '1', '218', '10cc___Im_Not_In_Love'],
        ]
        with open(self.output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)
        # test 2: already proccessed csv as output -- should be the same
        index_lyrics.index_lyrics(self.output_csv, self.output_csv)
        with open(self.output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)
        # test 3: input csv with mixed processed & unprocessed rows -- should process remaining rows
        with open(self.output_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['10cc', "Woman In Love", '0', '-1', '-1', 'aaa'])
            writer.writerow(['10cc', "The Things We Do For Love", '-1', '-1', '111', ''])
            writer.writerow(['apple', "orange", '-1', '-1', 'aaa', ''])
        expected_rows.append(['10cc', "Woman In Love", '1', '1', '300', '10cc___Woman_In_Love'])
        expected_rows.append(['10cc', "The Things We Do For Love", '1', '1', '286', '10cc___The_Things_We_Do_For_Love'])
        expected_rows.append(['apple', "orange", '0', '0', '0', 'apple___orange'])
        index_lyrics.index_lyrics(self.output_csv, self.output_csv)
        with open(self.output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)

                
class TestLabelLyrics(unittest.TestCase):

    def test_match_song_tags_to_mood_expanded(self):
        """
        Uses the moods "aggression", "angst", "brooding" to validate that
        expanded mood matching works as expected
        
        For reference, these moods were pasted in on 11/26/18
            "aggression": [
                ["aggression", "aggressive"],
                ["aggress"],
                ["not so aggressive"]
            ],
            "angst": [
                ["angst", "anxiety", "anxious", "jumpy", "nervous", "angsty"],
                ["angst", "anxiety", "anxious", "jumpy", "nervous", "angsty"],
                ["gangst", "langstrumpf", "farangstar", "gaaangstaa", "klangstark"]
            ],
            "brooding": [
                ["brooding", "contemplative", "meditative", "reflective"],
                ["brood", "contemplat", "meditat", "reflect"],
                ["broodcast", "marilyn manson - the reflecting god", "silverchair-reflections of a sound"]
            ],
        """
        tags = list()
        # should match aggressive 2
        tags += ['aggressiiiiive', 'not so aggressive', 'not so aggressss']
        # should match angst 2
        tags += ['jumpy', 'nervou', 'nervous', 'gangst', 'langstrumpf']
        # should match brooding 3
        tags += ['marilyn manson', 'broodccast', 'broodcast', 'contemplative', 'meditation']
        expected_mood = 'brooding'
        expected_scoreboard = dict.fromkeys(label_lyrics.MOOD_CATEGORIES_EXPANDED.keys(), 0)
        expected_scoreboard['aggression'] = 2
        expected_scoreboard['angst'] = 2
        expected_scoreboard['brooding'] = 3
        actual_mood, actual_scoreboard = label_lyrics.match_song_tags_to_mood_expanded(pd.Series(tags))
        self.assertEqual(expected_mood, actual_mood)
        self.assertEqual(expected_scoreboard, actual_scoreboard)


class TestLyrics2Vec(unittest.TestCase):
    
    #def lyrics_preprocessing(lyrics):
    def test_lyrics_preprocessing(self):
        lyrics = ("[Verse 1] "
                  "Yesterday All my troubles seemed so far away "
                  "Now it looks as though they're here to stay "
                  "Oh, I believe in yesterday "
                  "[Verse 2] "
                  "Suddenly "
                  "I'm not half the man I used to be "
                  "There's a shadow hanging over me "
                  "Oh, yesterday came suddenly")
        expected = ['verse', '1',
                    'yesterday', 'all', 'my', 'troubles', 'seemed', 'so', 'far', 'away',
                    'now', 'it', 'looks', 'as', 'though', 'they', "'re", 'here', 'to', 'stay',
                    'oh', 'i', 'believe', 'in', 'yesterday',
                    'verse', '2',
                    'suddenly',
                    'i', "'m", 'not', 'half', 'the', 'man', 'i', 'used', 'to', 'be',
                    'there', "'s", 'a', 'shadow', 'hanging', 'over', 'me',
                    'oh', 'yesterday', 'came', 'suddenly']
        actual = lyrics2vec.lyrics_preprocessing(lyrics)
        self.assertEqual(expected, actual)


class TestLyricsCnnDataImport(unittest.TestCase):

    test_txt = 'test.txt'
    input_csv = 'test_input.csv'
    output_csv = 'test_output.csv'

    def tearDown(self):
        if os.path.exists(self.input_csv):
            os.remove(self.input_csv)
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def test_import_labeled_lyrics_data(self):

        orig_cols = lyrics_cnn.LABELED_LYRICS_KEEP_COLS
        extra_cols = ['a', 'b', 'c']
        cols = orig_cols + extra_cols
        with open(self.input_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerow(['test'] * len(cols))
        
        expected = pd.read_csv(self.input_csv, usecols=orig_cols)
        actual = lyrics_cnn.import_labeled_lyrics_data(self.input_csv)

        self.assertTrue(expected.equals(actual))
    
    def test_filter_labeled_lyrics_data(self):
        cols = lyrics_cnn.LABELED_LYRICS_KEEP_COLS
        rows = [1, 1, 1, 1]
        rows = list(map(lambda x: [0] * len(cols), rows))
        rows[3] = [1] * len(cols)  # the one that will be left behind
        rows[0][cols.index('is_english')] = 1
        rows[1][cols.index('lyrics_available')] = 1
        rows[2][cols.index('matched_mood')] = 1
        with open(self.input_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for row in rows:
                writer.writerow(row)
        df = pd.read_csv(self.input_csv)
        df = df.drop([0, 1, 2])
        # test 1: with drop
        expected_df = df.drop(['is_english', 'lyrics_available', 'matched_mood'], axis=1)
        df_with_drop = lyrics_cnn.filter_labeled_lyrics_data(df)
        self.assertTrue(expected_df.equals(df_with_drop))
        # test 2: without drop
        expected_df = df
        df_without_drop = lyrics_cnn.filter_labeled_lyrics_data(df, drop=False)
        self.assertTrue(expected_df.equals(df_without_drop))

    def test_categorize_labeled_lyrics_data(self):
        cols = lyrics_cnn.LABELED_LYRICS_KEEP_COLS
        rows = [1, 2, 3, 4, 5, 6, 7]
        rows = list(map(lambda x: [1] * len(cols), rows))
        rows[3] = [1] * len(cols)  # the one that will be left behind
        rows[0][cols.index('mood')] = 'cat'
        rows[1][cols.index('mood')] = 'dog'
        rows[2][cols.index('mood')] = 'monkey'
        rows[3][cols.index('mood')] = 'monkey'
        rows[4][cols.index('mood')] = 'cat'
        rows[5][cols.index('mood')] = 'dog'
        rows[6][cols.index('mood')] = 'monkey'
        with open(self.input_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for row in rows:
                writer.writerow(row)
        # test
        df = pd.read_csv(self.input_csv)
        expected = df.copy()
        expected['mood_cats'] = pd.Series([0, 1, 2, 2, 0, 1, 2])
        expected.mood = expected.mood.astype('category')
        expected.mood_cats = expected.mood_cats.astype(np.int8)
        actual = lyrics_cnn.categorize_labeled_lyrics_data(df)
        self.assertTrue(expected.equals(actual))
    
    def test_extract_lyrics(self):
        # test 1: file exists
        expected = 'testing123'
        with open(self.test_txt, 'w') as f:
            f.write(expected)
        actual = lyrics_cnn.extract_lyrics(self.test_txt)
        self.assertEqual(expected, actual)
        # test 2: file does not exist
        expected = ''
        actual = lyrics_cnn.extract_lyrics('not exist this file does')
        self.assertEqual(expected, actual)

    def test_make_lyrics_txt_path(self):
        expected = 'bbb/' + 'aaa' + '.txt'
        actual = lyrics_cnn.make_lyrics_txt_path('aaa', 'bbb')
        self.assertEqual(expected, actual)

    def test_split_data(self):
        # set seed so that output is guaranteed reproducible
        np.random.seed(12)
        df = pd.DataFrame({'a': [0,1,2,3,4,5,6,7,8,9]})
        # test to make sure that 60% are in 1, 20% in 2, 20% in 3
        exp1 = df.reindex([5,8,7,0,4,9])
        exp2 = df.reindex([3,2])
        exp3 = df.reindex([1,6])
        act1, act2, act3 = lyrics_cnn.split_data(df)
        # output of func with random seed 12
        #   a
        #5  5
        #8  8
        #7  7
        #0  0
        #4  4
        #9  9
        #   a
        #3  3
        #2  2
        #   a
        #1  1
        #6  6
        self.assertTrue(exp1.equals(act1))
        self.assertTrue(exp2.equals(act2))
        self.assertTrue(exp3.equals(act3))
        
class TestLyricsCnn(unittest.TestCase):
    
    def setUp(self):
        """
        Generated by copy pasting the init func and some nifty regexes
        """
        
        self.batch_size = 128
        self.num_epochs = 99999999  # make extra large so that when we delete
                                    # dir we don't accidentally delete data
        self.sequence_length = 100
        self.num_classes = 8
        self.vocab_size = 50000
        self.embedding_size = 300
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128
        self.l2_reg_lambda = .01
        self.dropout = .001
        self.pretrained_embeddings = None
        self.train_embeddings = False
        self.evaluate_every = 10
        self.checkpoint_every = 11
        self.num_checkpoints = 12
        
        self.cnn = lyrics_cnn.LyricsCNN(batch_size=self.batch_size,
                                        num_epochs=self.num_epochs,
                                        sequence_length=self.sequence_length,
                                        num_classes=self.num_classes,
                                        vocab_size=self.vocab_size,
                                        embedding_size=self.embedding_size,
                                        filter_sizes=self.filter_sizes,
                                        num_filters=self.num_filters,
                                        l2_reg_lambda=self.l2_reg_lambda,
                                        dropout=self.dropout,
                                        pretrained_embeddings=self.pretrained_embeddings,
                                        train_embeddings=self.train_embeddings,
                                        evaluate_every=self.evaluate_every,
                                        checkpoint_every=self.checkpoint_every,
                                        num_checkpoints=self.num_checkpoints)

        return
    
    def tearDown(self):
        # clear all tensors and ops and variables because setUp runs before each test
        tf.reset_default_graph()
    
    def test___init__(self):
        """
        Conveniently also tests some of the dimensions of the TF variables
        """
        self.assertEqual(self.cnn.batch_size, self.batch_size)
        self.assertEqual(self.cnn.num_epochs, self.num_epochs)
        self.assertEqual(self.cnn.sequence_length, self.sequence_length)
        self.assertEqual(self.cnn.num_classes, self.num_classes)
        self.assertEqual(self.cnn.vocab_size, self.vocab_size)
        self.assertEqual(self.cnn.embedding_size, self.embedding_size)
        self.assertEqual(self.cnn.filter_sizes, self.filter_sizes)
        self.assertEqual(self.cnn.num_filters, self.num_filters)
        self.assertEqual(self.cnn.l2_reg_lambda, self.l2_reg_lambda)
        self.assertEqual(self.cnn.dropout, self.dropout)
        self.assertEqual(self.cnn.pretrained_embeddings, self.pretrained_embeddings)
        self.assertEqual(self.cnn.train_embeddings, self.train_embeddings)
        self.assertEqual(self.cnn.evaluate_every, self.evaluate_every)
        self.assertEqual(self.cnn.checkpoint_every, self.checkpoint_every)
        self.assertEqual(self.cnn.num_checkpoints, self.num_checkpoints)

        self.assertEqual(self.cnn.experiment_name, self.cnn._build_experiment_name())
        self.assertEqual(self.cnn.output_dir, self.cnn._build_output_dir())
        
    def test__build_output_dir(self):
        # test 1: with output_dir
        root = 'test_tmp_dir'
        expected = os.path.abspath(root) + '/' + self.cnn.experiment_name
        actual1 = self.cnn._build_output_dir(output_dir=root)
        self.assertEqual(expected, actual1)
        # test 2: without output_dir
        expected = os.path.abspath(lyrics2vec.LOGS_TF_DIR) + '/runs/' + self.cnn.experiment_name
        actual2 = self.cnn._build_output_dir()
        self.assertEqual(expected, actual2)
        # tear down
        shutil.rmtree(actual1)
        self.assertFalse(os.path.exists(actual1))
        shutil.rmtree(actual2)
        self.assertFalse(os.path.exists(actual2))
        
    #def _build_experiment_name(self, timestamp=False):
    def test__build_experiment_name(self):
        # skip timestamp test - idk how to generate exact same timestamp
        # test 1: parameter-unique name
        expected = 'Em-300_FS-3-4-5_NF-128_D-0.001_L2-0.01_B-128_Ep-99999999_W2V-0_V-50000'
        actual = self.cnn._build_experiment_name()
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()