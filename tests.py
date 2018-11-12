# python packages
import os
import csv
import json
import unittest
import pandas as pd

# project files
import etl_lyrics


class TestETL(unittest.TestCase):

    def test_read_file_contents(self):
        testfile = 'test.txt'
        # txt
        expected_contents = 'hello\nworld\n123'
        with open(testfile, 'w') as f:
            f.write(expected_contents)
        actual_contents = etl_lyrics.read_file_contents(testfile)
        self.assertEqual(expected_contents, actual_contents)
        # json
        expected_contents = {'hello': 'world', '1': {'2': '3', '4': '5'}}
        with open(testfile, 'w') as f:
            json.dump(expected_contents, f)
        actual_contents = etl_lyrics.read_file_contents(testfile, read_json=True)
        self.assertEqual(expected_contents, actual_contents)
        os.remove(testfile)

    def test_add_col_if_dne(self):
        data = {'a': ['b', 'c'], 'd': ['e', 'f']}
        df = pd.DataFrame(data)
        expected_df = pd.DataFrame({'a': ['b', 'c'], 'd': ['e', 'f'], 'g': ['h', 'h']})
        actual_df = etl_lyrics.add_col_if_dne(df, 'g', 'h')  # should be added
        actual_df = etl_lyrics.add_col_if_dne(df, 'a', 'b')  # should not be added
        self.assertEqual(expected_df.to_dict(), actual_df.to_dict())

    def test_etl_lyrics(self):
        input_csv = 'test_input.csv'
        output_csv = 'test_output.csv'
        with open(input_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['msd_artist', 'msd_title'])
            writer.writerow(['10cc', "I'm not in love"])
        # test 1: vanilla input csv with no additional columns
        etl_lyrics.etl_lyrics(input_csv, output_csv)
        expected_rows = [
            ['msd_artist', 'msd_title', 'is_english', 'lyrics_available', 'lyrics_filename'],
            ['10cc', "I'm not in love", '1', '1', '10cc___Im_not_in_love'],
        ]
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)
        # test 2: already proccessed csv as output -- should be the same
        etl_lyrics.etl_lyrics(output_csv, output_csv)
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)
        # test 3: input csv with mixed processed & unprocessed rows -- should process remaining rows
        with open(input_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['10cc', "woman in love", '0', '-1', 'aaa'])
            writer.writerow(['10cc', "The things we do for love", '-1', '-1', ''])
            writer.writerow(['apple', "orange", '-1', '-1', ''])
        expected_rows.append(['10cc', "The things we do for love", '1', '1', '10cc___The_Things_We_Do_For_Love'])
        expected_rows.append(['10cc', "woman in love", '1', '1', '10cc___Woman_In_Love'])
        expected_rows.append(['apple', "orange", '0', '0', ''])
        etl_lyrics.etl_lyrics(output_csv, output_csv)
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            for index, actual_row in enumerate(reader):
                self.assertEqual(expected_rows[index], actual_row)
        # cleanup
        os.remove(input_csv)
        os.remove(output_csv)



if __name__ == '__main__':
    unittest.main()