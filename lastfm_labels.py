"""

Useful tutorial from MSD: https://labrosa.ee.columbia.edu/millionsong/sites/default/files/lastfm/demo_tags_db.py
"""
import os
import sys
import sqlite3
import pandas as pd


LASTFM_TAGS_DB = 'data/lastfm_tags.db'


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


def main():

    dbfile = LASTFM_TAGS_DB

    # sanity check
    if not os.path.isfile(dbfile):
        print('ERROR: db file {0} does not exist?'.format(dbfile))

    # open connection
    conn = sqlite3.connect(dbfile)

    # EXAMPLE 4
    print('************** DEMO 4 **************')
    tag = 'happy'
    print('We get all tracks for the tag: {0}'.format(tag))
    sql = "SELECT tids.tid FROM tid_tag, tids, tags WHERE tids.ROWID=tid_tag.tid AND tid_tag.tag=tags.ROWID AND tags.tag='%s'" % sanitize(tag)
    res = conn.execute(sql)
    data = res.fetchall()
    print(len(data))

if __name__ == '__main__':
    main()
