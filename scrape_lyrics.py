"""
This script iterates through the artist:song pairs in the musixmatch file and
uses the lyricgenius package to retrieve the lyrics for each song from
Genius.com. The lyrics are then saved as json and as txt files.

An additional file is created, no_lyrics.csv, to track which songs we are unable
to retrieve lyrics for. This file is later reused in subsequent runs of this
script to avoid wasting time attempting to download lyrics for a song we know
we cannot find. The same logic applies to songs we already have lyrics for. The
script will not attempt to find lyrics for songs that already exist in the data
directory.

If
"""
import lyricsgenius as genius
import pandas as pd
import datetime
import logging
import time
import csv
import os


FILE_MUSIXMATCH_MAPPING = 'data/musixmatch_matches/mxm_779k_matches.txt'
CSV_MUSIXMATCH_MAPPING = 'data/mxm_mappings.csv'
CSV_NO_LYRICS = 'data/no_lyrics.csv'
CSV_HEADER = ['msd_id', 'msd_artist', 'msd_title', 'mxm_id', 'mxm_artist', 'mxm_title']
SKIP_LYRIC_CHECK_IF_KNOWN_BAD = True


logger = logging.getLogger(__name__)


def configure_logging(verbosity=1):

    global logger

    level = logging.INFO
    if verbosity is not None and int(verbosity) > 0:
        level = logging.DEBUG

    logger.setLevel(logging.DEBUG)  # we adjust on console and file later
    # create file handler which logs even debug messages
    fh = logging.FileHandler(datetime.datetime.now().strftime('logs/scrape_lyrics__%Y-%m-%d_%H-%M.log'), 'w', 'utf-8')
    fh.setLevel(logging.DEBUG)  # always log everything to file
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)  # only log to console what the user wants
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return


def musixmatch_mapping_to_csv():
    """
    Converts the musixmatch mapping file to a csv format for easier consumption
    """

    # to save time, do not execute again if conversion already exists
    if os.path.exists(CSV_MUSIXMATCH_MAPPING):
        logger.info('{0} already exists. Skipping csv conversion.'.format(CSV_MUSIXMATCH_MAPPING))
        return

    if not os.path.exists(FILE_MUSIXMATCH_MAPPING):
        logger.error('{0} not found. Please run download_data.py first.'.format(FILE_MUSIXMATCH_MAPPING))
        return

    count = 0
    with open(CSV_MUSIXMATCH_MAPPING, mode='w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(CSV_HEADER)

        with open(FILE_MUSIXMATCH_MAPPING, mode='r', encoding="utf-8") as f:
            for line in f:
                # check for comments
                if line.startswith('#'):
                    continue
                # check file for more info on file format
                try:
                    split = line.strip().split('<SEP>')
                    if len(split) != len(header):
                        logger.warning('interesting line: {0}'.format(line))
                    csvwriter.writerow(split)
                    count += 1

                except Exception as exc:
                    logger.warning(exc)

    logger.info('Converted {0} songs from musixmatch mapping to csv.'.format(count))

    return


def get_api_token():
    """
    Retrieves the genius api token from the saved api txt file
    """
    api_token = None
    with open('data/api.txt', 'r') as f:
        api_token = f.read()

    return api_token


def make_lyric_file_name(artist, track):
    # https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    keepcharacters = ('.','_')
    filename = '{0}___{1}'.format(artist.replace(' ', '_'), track.replace(' ', '_'))
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


def scrape_lyrics():
    """
    Iterate through the musixmatch csv file and attempt to find the lyrics for each
    with the genius api service

    A csv is created with the artist-song pairs that this function fails to find lyrics
    for.
    """

    api_token = get_api_token()
    api = genius.Genius(client_access_token=api_token, verbose=False)

    df = pd.read_csv(CSV_MUSIXMATCH_MAPPING, encoding='utf-8')

    # sort by artist so that we can
    df = df.sort_values('msd_artist')
    # would it be more efficient to groupby artist name?
    # maybe, but you run the risk of missing a song if msd_artist is incorrect spelling

    if not os.path.exists(CSV_NO_LYRICS):
        with open(CSV_NO_LYRICS, mode='w', encoding='utf-8', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(CSV_HEADER)
    df_no_lyrics = pd.read_csv(CSV_NO_LYRICS, encoding='utf-8')
    df_no_lyrics = df_no_lyrics.sort_values('msd_artist')

    song_index = 0
    songs_skipped = 0
    songs_matched = 0

    try:
        for index, row in df.iterrows():

            song_index += 1

            json_lyricfile = 'data/lyrics/json/{0}.json'.format(make_lyric_file_name(row['msd_artist'], row['msd_title']))
            txt_lyricfile = 'data/lyrics/txt/{0}.txt'.format(make_lyric_file_name(row['msd_artist'], row['msd_title']))

            # https://stackoverflow.com/questions/24761133/pandas-check-if-row-exists-with-certain-values
            if SKIP_LYRIC_CHECK_IF_KNOWN_BAD and (df_no_lyrics == row).all(1).any():
                logger.debug('{0}: (artist={1}, title={2}) matched in {3}. Skipping.'.format(song_index, row['msd_artist'], row['mxm_artist'], CSV_NO_LYRICS))
                songs_skipped +=1
                continue

            if os.path.exists(json_lyricfile):
                logger.debug('{0}: {1} already downloaded. Skipping.'.format(song_index, json_lyricfile))
                songs_skipped +=1
                continue

            try:
                song = api.search_song(row['msd_title'], row['msd_artist'])

                if not song:
                    # lets try all of the combinations!
                    artist_mismatch = row['msd_artist'] != row['mxm_artist']
                    title_mismatch = row['msd_title'] != row['mxm_title']
                    if artist_mismatch:
                        song = api.search_song(row['msd_title'], row['mxm_artist'])
                    if not song and title_mismatch:
                        song = api.search_song(row['mxm_title'], row['msd_artist'])
                    if not song and artist_mismatch and title_mismatch:
                        song = api.search_song(row['mxm_title'], row['mxm_artist'])
                    if not song:
                        # no luck... on to the next one
                        # https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe/24287210
                        df_no_lyrics.loc[-1] = row
                        df_no_lyrics.index += 1
                        logger.debug('{0}: No luck (artist={1}, title={2}). Saved to no lyrics csv.'.format(song_index, row['msd_artist'], row['mxm_artist']))
                        continue

                songs_matched += 1

                # save_lyrics function: https://github.com/johnwmillr/LyricsGenius/blob/master/lyricsgenius/song.py
                song.save_lyrics(filename=json_lyricfile, overwrite=True, verbose=False, format_='json')
                song.save_lyrics(filename=txt_lyricfile, overwrite=True, verbose=False, format_='txt')
                logger.debug('{0}: Success! (artist={1}, title={2}) saved to {3}.'.format(song_index, row['msd_artist'], row['msd_title'], json_lyricfile))

            except Exception as exc:
                logger.warning('Problem: {0}'.format(songs_matched))
                logger.warning(row)
                logger.warning(exc)

    except KeyboardInterrupt as kbi:
        logger.info(kbi)
        logger.info('saving no lyrics csv...')
        df_no_lyrics.to_csv(CSV_NO_LYRICS, encoding='utf-8', index=False)
        logger.info('done.')

    logger.info('{0} / {1} Song Lyrics Obtained! ({2} skipped)'.format(songs_matched, song_index, songs_skipped))

    return


def main():

    configure_logging()
    musixmatch_mapping_to_csv()
    scrape_lyrics()


if __name__ == '__main__':
    main()
