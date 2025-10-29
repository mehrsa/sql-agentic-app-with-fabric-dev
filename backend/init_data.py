import pyodbc
from shared.connection_manager import connection_manager
import os

def check_and_ingest_data():
    """Check if data exists in the DB, if not, ingest it from the SQL script."""
    conn = None
    cursor = None
    try:
        # Use the shared connection manager
        conn = connection_manager.create_connection()
        cursor = conn.cursor()
        
        print("[0] Checking database for existing data...")
        try:
            # 1. Check if data exists in a key table
            cursor.execute("SELECT COUNT(*) FROM dbo.PDF_RawChunks")
            count = cursor.fetchone()[0]
        except pyodbc.Error as e:
            # This likely means the table doesn't exist yet
            print(f"    Info: Could not check table (may not exist yet). Proceeding with ingestion.")
            count = 0 
        
        if count > 0:
            print(f"    Data ({count} rows) found in PDF_RawChunks. Skipping ingestion.")
            return

        # 2. If no data, run ingest script
        print("    No data found. Starting data ingestion...")
        # Path is relative to this launcher.py file
        sql_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data_Ingest', 'ingest_data.sql')
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Split the script by "GO" statements (which pyodbc doesn't run in one go)
        sql_commands = sql_script.split('GO')
        
        for command in sql_commands:
            command = command.strip()
            if command:
                try:
                    cursor.execute(command)
                except pyodbc.Error as ex:
                    # Print errors but continue (e.g., "DROP TABLE" if not exists might fail)
                    print(f"    Warning executing command: {ex}")
        
        conn.commit()
        print("    Database ingestion complete!")

    except pyodbc.Error as ex:
        print(f"    DATABASE ERROR during ingestion: {ex}")
        if conn:
            conn.rollback()
    except FileNotFoundError:
        print(f"    FATAL ERROR: SQL ingest file not found at {sql_file_path}")
    except Exception as e:
        print(f"    An unexpected error occurred during data check/ingestion: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        # Don't close the connection - let the connection manager handle it