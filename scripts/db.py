"""Read-only local source connection; never copy or print credentials."""
import importlib.util
import os
from pathlib import Path
import psycopg2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def connection(database):
    if os.environ.get('PGHOST') or os.environ.get('PGSERVICE'):
        # libpq reads the user's environment/service configuration or .pgpass.
        conn = psycopg2.connect(dbname=database, connect_timeout=10)
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("set statement_timeout='600s'")
        return conn
    path = Path(r'E:\Physionet\analysis\critical_care_rare_topic_discovery_20260710\scripts\extract_mimic_landmark.py')
    spec = importlib.util.spec_from_file_location('_existing_local_auth', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    conn = psycopg2.connect(host=module.HOST, port=module.PORT, user=module.USER,
        password=module.resolve_password(), dbname=database, connect_timeout=10)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("set statement_timeout='600s'")
    return conn

def query(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
