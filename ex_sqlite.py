# This is a snippets for sqlite3. Don't run this script.

import sqlite3
import pandas as pd

df = pd.read_csv('tube_all.csv')
df = df.iloc[:30, :]

#conn = sqlite3.connect(':memory:')
conn = sqlite3.connect(database = 'tube.db')

c = conn.cursor()

c.execute("""CREATE TABLE tube (
            channel_id INTEGER,
            name TEXT,
            subscriberCount INTEGER);""")

c.execute("""CREATE TABLE videos (
            video_id INTEGER,
            video_title TEXT,
            viewcounts INTEGER,
            channel INTEGER,
            FOREIGN KEY(channel) REFERENCES tube(channel_id));""")

#conn.commit()     # No need if I use with clause

#c.execute("INSERT INTO employees VALUES ('Jiwon', 'Jeong', 5000)")

def save_in_db(row):
    with conn:
        c.execute("INSERT INTO tube VALUES (:channel_id, :name, :subscriberCount)",
                {'channel_id': row.channel_id, 'name': row.name, 'subscriberCount': row.chn_subscriberCount2})
        c.execute("INSERT INTO videos VALUES (:video_id, :video_title, :viewcounts, :channel)",
                {'video_id': row.video_id, 'video_title': row.title, 'viewcounts': row.viewCount, 'channel':row.channel_id})


def update_pay(data, num):
    with conn:
        c.execute("""UPDATE tube SET num = :subscriberCount WEHRE channel_id = :channel_id""",
                {'channel_id': data.channel_id, 'subscriberCount': data.chn_subscriberCount})


def search_by_channel_name(channel_id):
    c.execute("SELECT * FROM tube WHERE channel_id=:channel_id", {'channel_id':channel_id})
    return c.fetchmany(5)

for i in range(len(df)):
    data = df.iloc[i, :]
    save_in_db(data)

#update_pay(data)
#c.fetchall()

ex = search_by_channel_name('UC-i2ywiuvjvpTy2zW-tXfkw')
for i in ex:
    print(i)

conn.close()
