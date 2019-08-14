# This is a snippets for sqlite3. Don't run this script.

import sqlite3
import pandas as pd

df = pd.read_csv('blade_final.csv')

#conn = sqlite3.connect(':memory:')
conn = sqlite3.connect(database = 'tube.db')
c = conn.cursor()

def create_table():
    c.execute("""CREATE TABLE IF NOT EXISTS tube (channel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                  name TEXT,
                                                  subscriberCount INTEGER);""")

    c.execute("""CREATE TABLE videos (video_id INTEGER,
                                      video_title TEXT,
                                      viewcounts INTEGER,
                                      channel INTEGER,
                                      FOREIGN KEY(channel) REFERENCES tube(channel_id));""")
    conn.commit()     # No need if I use with clause

# create_table()
# c.execute("INSERT INTO tube VALUES ('Jiwon', 'Jeong', 5000)")

def save_in_db(idx):
    with conn:
        c.execute("INSERT INTO tube (name, subscriberCount) VALUES (df.name[i], df.chn_subs[i])")

        c.execute("INSERT INTO videos VALUES (:video_id, :video_title, :viewcounts, :channel)",
                {df.video_id[i], df.title[i], df.viewCount[i], df.channel_id[i]})


for idx in range(10):
    save_in_db(idx)

# def update_pay(data, num):
#     with conn:
#         c.execute("""UPDATE tube SET num = :subscriberCount WEHRE channel_id = :channel_id""",
#                 {'channel_id': data.channel_id, 'subscriberCount': data.chn_subscriberCount})
#
#
# def search_by_channel_name(channel_id):
#     c.execute("SELECT * FROM tube WHERE channel_id=:channel_id", {'channel_id':channel_id})
#     return c.fetchmany(5)

#update_pay(data)
#c.fetchall()

ex = search_by_channel_name('UC-i2ywiuvjvpTy2zW-tXfkw')
for i in ex:
    print(i)

conn.close()
