"""
This script iterates through the artist:song pairs in the musixmatch file and
uses the lyricgenius package to retrieve the lyrics for each song from
Genius.com. The lyrics are then saved as json files.
"""
import lyricsgenius as genius
import time
import csv
import os
# api = genius.Genius()
# artist = api.search_artist('Andy Shauf', max_songs=3)

EXAMPLE_URL = 'http://lyrics.wikia.com/wiki/Warren_Zevon:Werewolves_of_london'

FILE_MUSIXMATCH_MAPPING = 'data/musixmatch_matches/mxm_779k_matches.txt'
CSV_MUSIXMATCH_MAPPING = 'data/mxm_mappings.csv'
CSV_HEADER = ['msd_id', 'msd_artist', 'msd_title', 'mxm_id', 'mxm_artist', 'mxm_title']

def mapping_to_csv():

    # do not execute script again if lyrics have already been scraped
    if os.path.exists(CSV_MUSIXMATCH_MAPPING):
        print('{0} already exists. Skipping csv conversion.'.format(CSV_MUSIXMATCH_MAPPING))
        return

    count = 0
    with open(CSV_MUSIXMATCH_MAPPING, mode='w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)

        with open(FILE_MUSIXMATCH_MAPPING, mode='r', encoding="utf-8") as f:
            for line in f:
                # check for comments
                if line.startswith('#'):
                    continue
                # check file for more info on file format
                try:
                    split = line.strip().split('<SEP>')
                    if len(split) != len(header):
                        print('interesting line: {0}'.format(line))
                    csvwriter.writerow(split)
                    count += 1

                except Exception as exc:
                    print(exc)

    print('Processed {0} songs.'.format(count))

    return


def get_api_token():

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

    api_token = get_api_token()
    api = genius.Genius(client_access_token=api_token)

    songs_matched = 0
    with open(CSV_MUSIXMATCH_MAPPING, mode='r', encoding='utf-8', newline='') as csvfile:
        # would it be more efficient to groupby artist name?
        # maybe, but you run the risk of missing a song if msd_artist is incorrect spelling
        csvreader = csv.DictReader(csvfile, fieldnames=CSV_HEADER)
        first = True
        for row in csvreader:
            # skip header row
            if first:
                first = False
                continue

            song = api.search_song(row['msd_title'], row['msd_artist'])
            if song:
                songs_matched += 1
                lyricfile = 'data/lyrics/{0}'.format(make_lyric_file_name(row['msd_artist'], row['msd_title']))
                # save_lyrics function: https://github.com/johnwmillr/LyricsGenius/blob/master/lyricsgenius/song.py
                song.save_lyrics(filename=lyricfile, overwrite=True, verbose=False)

            if songs_matched == 10:
                break

    return


def main():

    mapping_to_csv()
    scrape_lyrics()


if __name__ == '__main__':
    main()
