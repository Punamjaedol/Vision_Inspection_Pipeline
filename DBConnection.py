import json
import mariadb
import sys
import pathlib as Path

_SETTINGS_PATH = Path(__file__).resolve().parent / "db_settings.json"

def _load_saved_settings():
    """이전에 GUI에서 저장해둔 접속정보가 있으면 기본값 위에 덮어씌운다."""
    global DB_HOST, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME
    if not _SETTINGS_PATH.exists():
        return
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        DB_HOST = data.get("host", DB_HOST)
        DB_USER = data.get("user", DB_USER)
        DB_PASSWORD = data.get("password", DB_PASSWORD)
        DB_PORT = int(data.get("port", DB_PORT))
        DB_NAME = data.get("dbname", DB_NAME)
    except Exception as e:
        print(f"[DB][SETTINGS] Failed to load saved settings: {e}")

_load_saved_settings()

def get_connection():
    """
    DB 연결 정보 반환 함수
    입력 : 없음
    출력 : MariaDB Connection 객체
    """
    return mariadb.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        database=DB_NAME
    )

def connect():
    """
    DB 연결
    입력 : 없음
    출력 : Connection 객체 연결 실패 시 None
    """
    try:
        conn = get_connection()
        # cursor = conn.cursor()
        print(f"[DB][CONNECT] {DB_HOST}:{DB_PORT} -> SUCCESS")

        return conn

    except mariadb.Error as e:
        print(f"[DB][DISCONNECT] Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
        return None
    
def close_connection(conn):
    if conn: conn.close()

# =====================
#       SQL Query
# =====================
def select(table, columns="*", where=None, params=None, extra=""):
    """
    SELECT * FROM table WHERE ...
    where 예: "id = %s AND name = %s"
    params 예: (1, "kim")
    """
    try:
        conn = connect()
        with conn.cursor() as cur:
            sql = f"SELECT {columns} FROM {table}"
            if where:
                sql += f" WHERE {where}"
            if extra:
                sql += f" {extra}"

            cur.execute(sql, params)
            result = cur.fetchall()

            print(f"[DB][SELECT] {table} -> {len(result)} rows")
            return result

    except mariadb.Error as e:
        print(f"[DB][SELECT][ERROR] {table} | {e}")
        return None
    finally:
        close_connection(conn)


def is_exist(table, where, params=None):
    try:
        conn = connect()
        with conn.cursor() as cur:
            sql = f"SELECT 1 FROM {table} WHERE {where} LIMIT 1"
            cur.execute(sql, params)
            result = cur.fetchone() is not None

            print(f"[DB][EXISTS] {table} -> {result}")    
            return result
    except mariadb.Error as e:
        print(f"[DB][SELECT][ERROR] {table} | {e}")
        return False
    finally:
        close_connection(conn)
    

def insert(table, data: dict):
    """
    data = {"col1": val1, "col2": val2}
    """
    try:
        conn = connect()

        cols = ",".join(data.keys())
        vals = ",".join(["%s"] * len(data))

        sql = f"INSERT INTO {table} ({cols}) VALUES ({vals})"
        
        with conn.cursor() as cur:
            cur.execute(sql, tuple(data.values()))
        conn.commit()

        print(f"[DB][INSERT] {table} -> SUCCESS")

    except mariadb.Error as e:
        print(f"[DB][INSERT][ERROR] {table} | {e}")
        sys.exit(1)
    finally:
        close_connection(conn)


def update(table, data: dict, where, params=None):
    """
    data = {"col1": val1}
    where = "id = %s"
    params = (id값,)
    """
    try:
        conn = connect()

        set_clause = ",".join([f"{k}=%s" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        values = tuple(data.values())

        if params:
            values = values + tuple(params)
        with conn.cursor() as cur:
            cur.execute(sql, values)
            affected_rows = cur.rowcount
        conn.commit()
        print(f"[DB][UPDATE] {table} -> {affected_rows} rows")
        
    except mariadb.Error as e:
        print(f"[DB][UPDATE][ERROR] {table} | {e}")
        print("[DB][UPDATE] ERROR:", e)
    finally:
        close_connection(conn)

