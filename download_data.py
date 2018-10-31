from zipfile import ZipFile
import requests
import json
import time
import os


DATA_DIR = 'data'


# dict - (datasrc, split, url, dest)
DATA_URLS = [
    # ('Last.fm', 'subset', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_subset.zip', 'data/lastfm_subset.json.zip'),
    # ('Last.fm', 'train', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_train.zip', 'data/lastfm_train.json.zip'),
    # ('Last.fm', 'test', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/lastfm_test.zip',  'data/lastfm_test.json.zip'),
    # ('musixmatch', 'train', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip', 'data/musixmatch_train.txt.zip'),
    # ('musixmatch', 'test', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip', 'data/musixmatch_test.txt.zip'),
    ('musixmatch', 'matches', 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_779k_matches.txt.zip', 'data/musixmatch_matches.txt.zip')
]


def payload_size(url):
    response = requests.head(url)
    return response.headers.get('content-length', None)


def download_zip(url, dest):
    response = requests.get(url)
    with open(dest, 'wb') as f:
        f.write(response.content)
    return


def unzip(zipfile_path, dest):
    # https://stackoverflow.com/questions/3451111/unzipping-files-in-python/3451150
    with ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(dest)
    return


def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print('-----------------------------------------------')
    print('Downloading\n')

    for datasrc, split, url, dest in DATA_URLS:
        print('Starting download for {0} {1}'.format(datasrc, split))
        if os.path.exists(dest):
            print('\tDataset already downloaded at {0}. Skipping.'.format(dest))
        else:
            #size = payload_size(url)
            #if size:
            #    print('Downloading Last.fm {1}... file size = {2}'.format(size))
            start = time.time()
            download_zip(url, dest)
            end = time.time()
            elapsed_time = end - start
            print('Dataset downloaded to {0}. Elapsed Time = {1} secs.'.format(dest, elapsed_time))

    print('\n-----------------------------------------------')
    print('Unzipping\n')

    for datasrc, split, url, dest in DATA_URLS:
        zippedfile = dest
        dest = dest[:dest.index('.')]  # extracted files will be put inside of this directory
        print('Starting unzip for {0} {1}'.format(datasrc, split))
        if os.path.exists(dest):
            print('\tDataset already unzipped at {0}. Skipping.'.format(dest))
        else:
            start = time.time()
            unzip(zippedfile, dest)
            end = time.time()
            elapsed_time = end - start
            print('Dataset unzipped to {0}. Elapsed Time = {1} secs.'.format(dest, elapsed_time))

if __name__ == '__main__':
    main()
