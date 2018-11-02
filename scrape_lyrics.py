"""
This script iterates through the artist:song pairs in the musixmatch file and
uses the lyricgenius package to retrieve the lyrics for each song from
Genius.com. The lyrics are then saved as json files.
"""
import lyricsgenius as genius
import csv
import os
# api = genius.Genius()
# artist = api.search_artist('Andy Shauf', max_songs=3)

EXAMPLE_URL = 'http://lyrics.wikia.com/wiki/Warren_Zevon:Werewolves_of_london'

FILE_MUSIXMATCH_MAPPING = 'data/musixmatch_matches/mxm_779k_matches.txt'
CSV_MUSIXMATCH_MAPPING = 'data/mxm_mappings.csv'


def main():

    # do not execute script again if lyrics have already been scraped
    if os.path.exists(CSV_MUSIXMATCH_MAPPING):
        print('{0} already exists. Exiting...'.format(CSV_MUSIXMATCH_MAPPING))
        return

    count = 0
    with open(CSV_MUSIXMATCH_MAPPING, 'w', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['msd_id', 'msd_artist', 'msd_title', 'mxm_id', 'mxm_artist', 'mxm_title']
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

    print('Processed {0} songs'.format(count))

    return


if __name__ == '__main__':
    main()
