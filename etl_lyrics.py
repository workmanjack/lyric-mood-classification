from langdetect import detect
import argparse
import pandas as pd
import time
import json
import os
from scrape_lyrics import make_lyric_file_name, CSV_MUSIXMATCH_MAPPING, configure_logging, logger


CSV_ETL = 'data/etl_lyrics.csv'


def read_file_contents(path, read_json=False):
    contents = None
    with open(path, 'r') as f:
        if read_json:
            contents = json.load(f)
        else:
            contents = f.read()
    return contents


def add_col_if_dne(df, col, value):
    if col not in df.columns:
        logger.debug('Adding column "{0}" to lyrics dataframe'.format(col))
        df[col] = value
    return df


def etl_lyrics(csv_input, csv_output, artist_first_letter=None):

    logger.info('Reading in input csv {0}'.format(csv_input))

    start = time.time()

    df = pd.read_csv(csv_input, encoding='utf-8', dtype = {'msd_artist':str, 'msd_title': str})
    # if starting from mxm_mapping csv, we need to add additional cols
    df = add_col_if_dne(df, 'is_english', -1)
    df = add_col_if_dne(df, 'lyrics_available', -1)
    df = add_col_if_dne(df, 'wordcount', -1)
    df = add_col_if_dne(df, 'lyrics_filename', -1)

    df = df.sort_values('msd_artist')

    end = time.time()
    elapsed_time = end - start

    logger.debug('Elapsed Time: {0} minutes'.format(elapsed_time / 60))

    logger.info('Processing Lyrics...')

    count_total = 0
    count_nonenglish = 0
    count_nolyrics = 0
    count_success = 0

    start = time.time()

    try:

        for index, row in df.iterrows():

            lyrics_filename = make_lyric_file_name(row['msd_artist'], row['msd_title'])
            txt_lyricfile = 'data/lyrics/txt/{0}.txt'.format(lyrics_filename)

            if artist_first_letter and not row['msd_artist'].lower().startswith(artist_first_letter.lower()):
                continue

            if row.get('lyrics_available', None) >= 0:
                logger.debug('{0}, {1}: already processed, skipping'.format(count_total, txt_lyricfile))
                continue

            wordcount = 0
            lyrics_available = 0
            is_english = 0

            if not os.path.exists(txt_lyricfile):

                logger.debug('{0}, {1}: no lyric file'.format(count_total, txt_lyricfile))

            else:

                lyrics_available = 1
                contents = read_file_contents(txt_lyricfile)

                if contents:

                    # drop the non-english
                    # possible speed improvement available via pandas:
                    # https://stackoverflow.com/questions/49261711/detecting-language-of-a-text-document-other-than-using-iterrows
                    lang = detect(str(contents))

                    is_english = 1 if lang == 'en' else 0
                    if not is_english:

                        logger.debug('{0}, {1}: not english'.format(count_total, txt_lyricfile))
                        count_nonenglish += 1

                    wordcount = len(contents.split())

                    # success!
                    logger.debug('{0}, {1}: success'.format(count_total, txt_lyricfile))
                    count_success += 1

            df.loc[index, 'lyrics_filename'] = lyrics_filename
            df.loc[index, 'lyrics_available'] = lyrics_available
            df.loc[index, 'wordcount'] = wordcount
            df.loc[index, 'is_english'] = is_english

            count_total += 1

    except KeyboardInterrupt as kbi:
        logger.info(str(kbi))

    logger.info('saving etl data to {0}'.format(csv_output))
    df = df.sort_values('msd_artist')
    df.to_csv(csv_output, encoding='utf-8', index=False)

    end = time.time()
    elapsed_time = end - start

    logger.info('{0} Artist/pairs processed'.format(count_total))
    logger.info('{0} deemed not english'.format(count_nonenglish))
    logger.info('{0} lacking lyrics'.format(count_total))
    logger.info('{0} songs ready'.format(count_success))

    logger.info('Elapsed Time: {0} minutes'.format(elapsed_time / 60))

    return


def parse_args():

    # parse args
    parser = argparse.ArgumentParser()

    # universal args
    parser.add_argument('-a', '--artist-first-letter', action='store', required=False, default=None, help='Attempt to download lyrics only for artists that start with this letter.')
    parser.add_argument('-i', '--csv-input', action='store', required=False, default=CSV_MUSIXMATCH_MAPPING, help='Artist-Song mapping csv')
    parser.add_argument('-o', '--csv-output', action='store', required=False, default=CSV_ETL, help='csv to write to (WARNING: will overwite)')

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

    configure_logging(logname='etl_lyrics')
    args = parse_args()
    etl_lyrics(args.csv_input, args.csv_output, args.artist_first_letter)

    return


if __name__ == '__main__':
    main()
