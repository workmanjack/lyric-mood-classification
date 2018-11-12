"""

Useful tutorial from MSD: https://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/demo_tags_db.py
"""
import os
import sys
import time
import sqlite3
import argparse
import pandas as pd
from scrape_lyrics import configure_logging, logger
from index_lyrics import CSV_INDEX_LYRICS, add_col_if_dne


CSV_LABELED_LYRICS = 'data/labeled_lyrics.csv'
LASTFM_TAGS_DB = 'data/lastfm_tags.db'
MOOD_UNKNOWN_KEY = 'unknown'


# from Table 2 on pg 413 of http://www.ismir2009.ismir.net/proceedings/PS3-4.pdf
MOOD_CATEGORIES = {
    'calm': ['calm', 'comfort', 'quiet', 'serene', 'mellow', 'chill out'],
    'sad': ['sadness', 'unhappy', 'melancholic', 'melancholy'],
    'happy': ['happy', 'happiness', 'happy songs', 'happy music'],
    'romantic': ['romantic', 'romantic music'],
    'upbeat': ['upbeat', 'gleeful', 'high spirits', 'zest', 'enthusiastic'],
    'depressed': ['depressed', 'blue', 'dark', 'depressive', 'dreary'],
    'anger': ['anger', 'angry', 'choleric', 'fury', 'outraged', 'rage'],
    'grief': ['grief', 'heartbreak', 'mournful', 'sorrow', 'sorry'],
    'dreamy': ['dreamy'],
    'cheerful': ['cheerful', 'cheer up', 'festive', 'jolly', 'jovial', 'merry'],
    'brooding': ['brooding', 'contemplative', 'meditative', 'reflective'],
    'aggression': ['aggression', 'aggressive'],
    'confident': ['confident', 'encouraging', 'encouragement', 'optimism'],
    'angst': ['angst', 'anxiety', 'anxious', 'jumpy', 'nervous', 'angsty'],
    'earnest': ['earnest', 'heartfelt'],
    'desire': ['desire', 'hope', 'hopeful', 'mood: hopeful'],
    'pessimism': ['pessimism', 'cynical', 'pessimistic', 'weltschmerz'],
    'excitement': ['excitement', 'exciting', 'exhilarating', 'thrill', 'ardor']
}


def sanitize(tag):
    """
    sanitize a tag so it can be included or queried in the db
    """
    tag = tag.replace("'","''")
    return tag


def query(db_conn, sql):
    """
    Execute the provided sql query on the provided db.

    WARNING: can cause memory issues
    """
    response = db_conn.execute(sql)
    data = response.fetchall()
    return data


def match_tag_to_mood(tag):
    matched_mood = MOOD_UNKNOWN_KEY
    for mood, submoods in MOOD_CATEGORIES.items():
        if tag in submoods:
            matched_mood = mood
            break
    return matched_mood


def label_lyrics(csv_input, csv_output, artist_first_letter=None):

    logger.info('Reading in input csv {0}'.format(csv_input))

    start = time.time()

    df = pd.read_csv(csv_input, encoding='utf-8', dtype = {'msd_artist':str, 'msd_title': str})
    df = add_col_if_dne(df, 'mood', '')
    df = add_col_if_dne(df, 'found_tags', -1)
    df = add_col_if_dne(df, 'matched_mood', -1)

    ## Doing this will drop all filtered songs from final csv but grants speed
    logger.debug('Filtering to only songs that are in english and have lyrics available')
    logger.debug('Songs before filtering: {0}'.format(len(df)))
    dropped_df = df[df['is_english'] != 1]
    df = df[df['is_english'] == 1]
    logger.debug('Songs after english filter: {0}'.format(len(df)))
    dropped_df = pd.concat([dropped_df, df[df['lyrics_available'] != 1]])
    df = df[df['lyrics_available'] == 1]
    logger.debug('Songs after lyric availability fliter: {0}'.format(len(df)))

    end = time.time()
    elapsed_time = end - start

    logger.debug('Elapsed Time: {0} minutes'.format(elapsed_time / 60))

    logger.info('Connecting to last.fm sqlite db at {0}'.format(LASTFM_TAGS_DB))

    dbfile = LASTFM_TAGS_DB

    # sanity check
    if not os.path.isfile(dbfile):
        print('ERROR: db file {0} does not exist? Try running download_data.py!'.format(dbfile))

    # open connection
    conn = sqlite3.connect(dbfile)

    logger.info('Querying Last.fm and Labeling Lyrics')

    count_total = 0
    count_lyrics_with_tags = 0
    count_no_mood_match = 0
    count_labeled = 0

    start = time.time()

    try:

        for index, row in df.iterrows():

            found_tags = 0
            matched_mood = 0
            match = MOOD_UNKNOWN_KEY

            # query for tags
            sql = "SELECT tids.tid, tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tids.ROWID=tid_tag.tid AND tid_tag.tag=tags.ROWID AND tids.tid='{0}'".format(sanitize(row['msd_id']))
            data = pd.read_sql_query(sql, conn)

            # check if tags were returned
            found_tags = len(data)
            if found_tags == 0:
                logger.debug('{0}, {1}: no tags'.format(count_total, row['lyrics_filename']))

            else:
                count_lyrics_with_tags += 1

                # attempt to match tag to a mood
                for tag in data['tag']:
                    match = match_tag_to_mood(tag)
                    if match != MOOD_UNKNOWN_KEY:
                        break

                # process the match, if any
                if match == MOOD_UNKNOWN_KEY:
                    logger.debug('{0}, {1}: found tags but could not match mood'.format(count_total, row['lyrics_filename']))
                    count_no_mood_match += 1

                else:
                    logger.debug('{0}, {1}: success! mood={2}'.format(count_total, row['lyrics_filename'], match))
                    matched_mood = 1
                    count_labeled += 1

            count_total += 1
            df.loc[index, 'found_tags'] = found_tags
            df.loc[index, 'matched_mood'] = matched_mood
            df.loc[index, 'mood'] = match

    except KeyboardInterrupt as kbi:
        logger.info(str(kbi))

    logger.debug('Merging all no-lyric rows to df...')
    logger.debug('Rows before merge (lyrics): {0}'.format(len(df)))
    logger.debug('Rows before merge (no lyrics): {0}'.format(len(dropped_df)))
    df = pd.concat([df, dropped_df])
    logger.debug('Rows after merge: {0}'.format(len(df)))
    logger.info('saving labeled lyric data to {0}'.format(csv_output))
    df.to_csv(csv_output, encoding='utf-8', index=False)

    end = time.time()
    elapsed_time = end - start

    logger.info('{0} songs processed'.format(count_total))
    logger.info('{0} songs with tags'.format(count_lyrics_with_tags))
    logger.info('{0} songs with no mood match'.format(count_no_mood_match))
    logger.info('{0} songs labeled'.format(count_labeled))

    logger.info('Elapsed Time: {0} minutes'.format(elapsed_time / 60))

    return


def parse_args():

    # parse args
    parser = argparse.ArgumentParser()

    # universal args
    parser.add_argument('-a', '--artist-first-letter', action='store', required=False, default=None, help='Attempt to label lyrics only for artists that start with this letter.')
    parser.add_argument('-i', '--csv-input', action='store', required=False, default=CSV_INDEX_LYRICS, help='Artist-Song mapping csv with lyric file paths')
    parser.add_argument('-o', '--csv-output', action='store', required=False, default=CSV_LABELED_LYRICS, help='csv to write to (WARNING: will overwite)')

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

    configure_logging(logname='label_lyrics')
    args = parse_args()
    label_lyrics(args.csv_input, args.csv_output, args.artist_first_letter)

    return

if __name__ == '__main__':
    main()
