import pandas as pd
import os
import argparse
import time
from scrape_lyrics import configure_logging, logger
from label_lyrics import CSV_LABELED_LYRICS

CSV_LABELED_GENRE = 'data/labeled_genre.csv'
GENRE_FILE = 'data/genres.csv'

def merge_genres(csv_input, genre_file, csv_output):
    start = time.time()

    logger.debug('Reading genre file...')
    df_genre = pd.read_csv(genre_file, names = ['msd_id', 'magd_genre'], sep = '\t')

    logger.debug('Reading lyrics file...')
    df_labeled = pd.read_csv(csv_input)

    logger.debug('Merging.')
    final_df = pd.merge(df_labeled, df_genre, how = 'left', on='msd_id')
    final_df.to_csv(csv_output, encoding='utf-8', index=False)

    elapsed_time = time.time() - start
    logger.debug('Elapsted Time: {0} minutes'.format(elapsed_time / 60))


def parse_args():

    # parse args
    parser = argparse.ArgumentParser()

    # universal args
    parser.add_argument('-g', '--genre_file', action='store', required=False, default=GENRE_FILE, help='File with msd_id, genre tab delimited')
    parser.add_argument('-i', '--csv-input', action='store', required=False, default=CSV_LABELED_LYRICS, help='Artist-Song mapping csv with lyric file paths')
    parser.add_argument('-o', '--csv-output', action='store', required=False, default=CSV_LABELED_GENRE, help='csv to write to (WARNING: will overwite)')

    args = parser.parse_args()

    # simple arg sanity check
    if os.path.exists(args.csv_output):
        logger.warning('Output CSV "{0}" will be overwritten'.format(args.csv_output))
        while True:
            response = input('Is this okay? Please enter Y/N:').lower()
            if response == 'y':
                break
            elif response == 'n':
                logger.info('exiting')
                import sys
                sys.exit(0)
            else:
                print('that is not an option')

    return args

def main():

    configure_logging(logname='merge_genres')
    args = parse_args()
    merge_genres(args.csv_input, args.genre_file, args.csv_output)

    return


if __name__ == '__main__':
    main()


