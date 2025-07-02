import psycopg2

# Database connection parameters
DB_USER = 'postgres'
DB_PASSWORD = 'gekko2022'
DB_HOST = 'host'
DB_PORT = '25432'
DB_NAME = 'monitoring'


# Connection to the DB
CONNECTION = f"postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor()

    # use the cursor to interact with your database
cursor.execute("SELECT 'hello world'")

query_create_sensors_table = """CREATE TABLE sensors (
                                    id SERIAL PRIMARY KEY,
                                    type VARCHAR(50),
                                    location VARCHAR(50)
                                );
                                """

cursor = conn.cursor()
# see definition in Step 1y
cursor.execute(query_create_sensors_table)
conn.commit()
cursor.close()


import pandas as pd
data = pd.read_csv("../DATA/store_data/BCFT.csv", header=0, decimal=",", sep=";")

from sqlalchemy import create_engine
# Create SQLAlchemy engine
DB_URL = f'postgresql://postgres:gekko2022@host:{25432}/monitoring'
engine = create_engine(DB_URL)

# Convert DataFrame to PostgreSQL table
table_name = 'test'
data.to_sql(table_name, engine, if_exists='replace', index=False)

# Close the database connection
engine.dispose()

query_create_sensordata_table = """CREATE TABLE sensor_data (
                                        time TIMESTAMPTZ NOT NULL,
                                        sensor_id INTEGER,
                                        temperature DOUBLE PRECISION,
                                        cpu DOUBLE PRECISION,
                                        FOREIGN KEY (sensor_id) REFERENCES sensors (id)
                                    );
                                    """