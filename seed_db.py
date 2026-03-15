"""
Seed a local SQLite database with sample data for the research agent.

Run once before using db_agent.py:
  python seed_db.py

Creates research.db with three tables: movies, countries, stocks.
"""

import sqlite3
from pathlib import Path

DB_PATH = "research.db"


def seed():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Movies ─────────────────────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS movies")
    cur.execute("""
        CREATE TABLE movies (
            id      INTEGER PRIMARY KEY,
            title   TEXT NOT NULL,
            year    INTEGER,
            genre   TEXT,
            rating  REAL,   -- IMDb-style 0–10
            director TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO movies (title, year, genre, rating, director) VALUES (?,?,?,?,?)",
        [
            ("The Godfather",           1972, "Crime",     9.2, "Francis Ford Coppola"),
            ("The Dark Knight",         2008, "Action",    9.0, "Christopher Nolan"),
            ("Schindler's List",        1993, "Drama",     8.9, "Steven Spielberg"),
            ("Pulp Fiction",            1994, "Crime",     8.9, "Quentin Tarantino"),
            ("Forrest Gump",            1994, "Drama",     8.8, "Robert Zemeckis"),
            ("Inception",               2010, "Sci-Fi",    8.8, "Christopher Nolan"),
            ("The Matrix",              1999, "Sci-Fi",    8.7, "The Wachowskis"),
            ("Goodfellas",              1990, "Crime",     8.7, "Martin Scorsese"),
            ("Interstellar",            2014, "Sci-Fi",    8.6, "Christopher Nolan"),
            ("The Silence of the Lambs",1991, "Thriller",  8.6, "Jonathan Demme"),
            ("Parasite",                2019, "Thriller",  8.5, "Bong Joon-ho"),
            ("The Lion King",           1994, "Animation", 8.5, "Roger Allers"),
            ("Whiplash",                2014, "Drama",     8.5, "Damien Chazelle"),
            ("The Grand Budapest Hotel",2014, "Comedy",    8.1, "Wes Anderson"),
            ("La La Land",              2016, "Romance",   8.0, "Damien Chazelle"),
            ("Get Out",                 2017, "Horror",    7.7, "Jordan Peele"),
            ("Everything Everywhere",   2022, "Sci-Fi",    7.8, "Daniels"),
            ("Oppenheimer",             2023, "Drama",     8.3, "Christopher Nolan"),
            ("Barbie",                  2023, "Comedy",    6.9, "Greta Gerwig"),
            ("Dune: Part Two",          2024, "Sci-Fi",    8.5, "Denis Villeneuve"),
        ],
    )

    # ── Countries ──────────────────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS countries")
    cur.execute("""
        CREATE TABLE countries (
            id                  INTEGER PRIMARY KEY,
            name                TEXT NOT NULL,
            continent           TEXT,
            population_millions REAL,
            gdp_billions        REAL,   -- USD, approximate
            capital             TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO countries (name, continent, population_millions, gdp_billions, capital) VALUES (?,?,?,?,?)",
        [
            ("United States",   "North America",  335.0,  27360.0, "Washington D.C."),
            ("China",           "Asia",          1410.0,  17960.0, "Beijing"),
            ("Germany",         "Europe",          84.0,   4260.0, "Berlin"),
            ("Japan",           "Asia",           125.0,   4230.0, "Tokyo"),
            ("India",           "Asia",          1428.0,   3730.0, "New Delhi"),
            ("United Kingdom",  "Europe",          67.0,   3090.0, "London"),
            ("France",          "Europe",          68.0,   3010.0, "Paris"),
            ("Brazil",          "South America",  215.0,   2080.0, "Brasília"),
            ("Canada",          "North America",   40.0,   2140.0, "Ottawa"),
            ("Italy",           "Europe",          59.0,   2170.0, "Rome"),
            ("Australia",       "Oceania",         26.0,   1690.0, "Canberra"),
            ("South Korea",     "Asia",            52.0,   1710.0, "Seoul"),
            ("Mexico",          "North America",  129.0,   1320.0, "Mexico City"),
            ("Netherlands",     "Europe",          17.7,   1090.0, "Amsterdam"),
            ("Argentina",       "South America",   46.0,    620.0, "Buenos Aires"),
            ("Nigeria",         "Africa",         223.0,    477.0, "Abuja"),
            ("Egypt",           "Africa",         105.0,    396.0, "Cairo"),
            ("South Africa",    "Africa",          60.0,    378.0, "Pretoria"),
            ("Norway",          "Europe",           5.5,    547.0, "Oslo"),
            ("New Zealand",     "Oceania",          5.1,    249.0, "Wellington"),
        ],
    )

    # ── Stocks ─────────────────────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS stocks")
    cur.execute("""
        CREATE TABLE stocks (
            id                  INTEGER PRIMARY KEY,
            ticker              TEXT NOT NULL,
            company             TEXT,
            sector              TEXT,
            price               REAL,   -- USD
            market_cap_billions REAL,
            pe_ratio            REAL    -- price / earnings ratio
        )
    """)
    cur.executemany(
        "INSERT INTO stocks (ticker, company, sector, price, market_cap_billions, pe_ratio) VALUES (?,?,?,?,?,?)",
        [
            ("AAPL",  "Apple Inc.",               "Technology",   189.3, 2940.0, 31.2),
            ("MSFT",  "Microsoft Corp.",           "Technology",   415.5, 3090.0, 36.8),
            ("GOOGL", "Alphabet Inc.",             "Technology",   175.0, 2180.0, 27.4),
            ("AMZN",  "Amazon.com Inc.",           "Consumer",     185.6, 1940.0, 61.3),
            ("NVDA",  "NVIDIA Corp.",              "Technology",   875.4, 2160.0, 68.5),
            ("META",  "Meta Platforms",            "Technology",   504.2, 1280.0, 28.9),
            ("TSLA",  "Tesla Inc.",                "Automotive",   177.9,  568.0, 45.2),
            ("BRK.B", "Berkshire Hathaway B",      "Finance",      388.5,  852.0, 21.3),
            ("JPM",   "JPMorgan Chase",            "Finance",      196.3,  569.0, 12.1),
            ("V",     "Visa Inc.",                 "Finance",      272.8,  560.0, 30.7),
            ("JNJ",   "Johnson & Johnson",         "Healthcare",   157.2,  378.0, 15.8),
            ("UNH",   "UnitedHealth Group",        "Healthcare",   520.1,  480.0, 22.4),
            ("XOM",   "ExxonMobil",                "Energy",       110.6,  446.0, 13.9),
            ("WMT",   "Walmart Inc.",              "Consumer",      67.2,  542.0, 28.6),
            ("PG",    "Procter & Gamble",          "Consumer",     163.8,  386.0, 25.9),
            ("HD",    "Home Depot",                "Retail",       355.6,  353.0, 22.8),
            ("DIS",   "Walt Disney Co.",           "Entertainment", 91.2,  167.0, 74.1),
            ("NFLX",  "Netflix Inc.",              "Entertainment",625.0,  271.0, 50.3),
            ("PYPL",  "PayPal Holdings",           "Finance",       62.1,   67.0,  9.8),
            ("INTC",  "Intel Corp.",               "Technology",    30.5,  129.0,  8.2),
        ],
    )

    conn.commit()
    conn.close()

    print(f"Database seeded: {Path(DB_PATH).resolve()}")
    print("Tables: movies (20 rows), countries (20 rows), stocks (20 rows)")


if __name__ == "__main__":
    seed()
