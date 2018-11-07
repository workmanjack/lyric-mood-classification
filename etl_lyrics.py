from langdetect import detect
import pandas as pd
import time
import os
from scrape_lyrics import make_lyric_file_name, CSV_MUSIXMATCH_MAPPING, configure_logging, logger


CSV_ETL = 'data/etl_lyrics.csv'


def read_file_contents(path):
    contents = None
    with open(path, 'r') as f:
        contents = f.read()
    return contents


def main():

    configure_logging(logname='etl_lyrics')

    df = pd.read_csv(CSV_MUSIXMATCH_MAPPING, encoding='utf-8', dtype = {'msd_artist':str, 'msd_title': str})
    df = df.sort_values('msd_artist')

    df['lyrics'] = ''
    count_total = 0
    count_nonenglish = 0
    count_nolyrics = 0
    count_added = 0

    start = time.time()

    try:

        for index, row in df.iterrows():
            txt_lyricfile = 'data/lyrics/txt/{0}.txt'.format(make_lyric_file_name(row['msd_artist'], row['msd_title']))
            # print(txt_lyricfile)
            if os.path.exists(txt_lyricfile):
                contents = read_file_contents(txt_lyricfile)
                if contents:
                    # drop the non-english
                    # possible speed improvement available via pandas:
                    # https://stackoverflow.com/questions/49261711/detecting-language-of-a-text-document-other-than-using-iterrows
                    lang = detect(str(contents))
                    if lang != 'en':
                        logger.debug('{0}, {1}: dropping -> not english'.format(count_total, txt_lyricfile))
                        df.drop(index, inplace=True)
                        count_nonenglish += 1
                    # save the lyrics
                    df.loc[index, 'lyrics'] = contents
                    logger.debug('{0}, {1}: success'.format(count_total, txt_lyricfile))
                    count_added += 1
                else:
                    # no lyrics so we drop it
                    logger.debug('{0}, {1}: dropping -> no lyrics'.format(count_total, txt_lyricfile))
                    df.drop(index, inplace=True)
                    count_nolyrics += 1
            count_total += 1
            # if count_total > 100:
            #     break

    except KeyboardInterrupt as kbi:
        logger.info(kbi)

    logger.info('saving etl csv...')
    df.to_csv(CSV_ETL, encoding='utf-8', index=False)
    logger.info('done.')

    end = time.time()
    elapsed_time = end - start

    logger.info('{0} Artist/pairs processed'.format(count_total))
    logger.info('{0} deemed not english'.format(count_nonenglish))
    logger.info('{0} lacking lyrics'.format(count_total))
    logger.info('{0} songs added'.format(count_added))

    logger.info('Elapsed Time: {0} minutes'.format(elapsed_time / 60))


if __name__ == '__main__':
    main()
