import sqlite3
def connectdb():
    conn = sqlite3.connect("data/fin_set.db")#连接到db
    return conn
