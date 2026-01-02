"""Create a sqlite database to store the video metadata"""
import sqlite3
from sqlite3 import Error
from typing import Optional, List, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)

class VideoMetadataDB:
    """Video metadata database manager with resource management"""
    def __init__(self, db_file: str) -> None:
        self.db_file = db_file
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_file)
            # return rows as dictionaries for easier access
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite version: {sqlite3.version_info}")
        except Error as ex:
            logger.error(f"Error connecting to database: {ex}")
            self.conn = None

    def create_table(self) -> None:
        """initialization of database scheme"""
        if not self.conn:
            return
        
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            language TEXT,
            is_generated BOOLEAN DEFAULT 0 
        );
        """
        try:
            with self.conn:  # automatically commits or roll back
                self.conn.execute(sql_create_table)
            logger.info("Video metadata table created successfully")
        except Error as ex:
            logger.error(f"Error creating table: {ex}")

    def insert_video_metadata(self, metadata: Tuple[str, str, str, bool]) -> Optional[int]:
        """
        Insert new video metadata into the table
        Note: We use '?' placeholders to prevent SQL injection
        """
        if not self.conn:
            return None
        
        sql = ''' INSERT INTO videos(video_id, title, language, is_generated)
                    VALUES(?,?,?,?) '''
        try:
            with self.conn:   # handles commit & rollback
                cur = self.conn.cursor()
                cur.execute(sql, metadata)
                # self.conn.commit()   # commit the changes
                return cur.lastrowid
        except sqlite3.IntegrityError:
            logger.warning(f"Error: video ID {metadata[0]} already exists")
        except Error as ex:
            logger.error(f"Error inserting data: {ex}")
        return None
        
    def select_all_rows(self) -> List[Optional[sqlite3.Row]]:
        """
        Query all rows in the videos table
        """
        if not self.conn:
            return []
        
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM videos")
            
            rows = cur.fetchall()

            print("\nCurrent videos in Database:")
            for row in rows:
                print(f"ID: {row['video_id']} | Title: {row['title']}")
            return rows
        except Error as ex:
            logger.error(f"Error fetching data: {ex}")
            return []
        
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    # database = "test.db"

    # conn = create_connection(database)

    # if conn is not None:
    #     # create table
    #     create_table(conn)

    #     # insert data
    #     video_1 = ("05gcl7CPxic", "Did this hiker find a PORTAL to another dimension?", "English", False)

    #     video_id1 = insert_video_metadata(conn, video_1)

    #     print(f"Inserted video ID: {video_id1}")

    #     select_all_videos(conn)

    #     # close connection
    #     conn.close()
    #     print("Database connection closed")

    # else:
    #     print("Error! cannot create database connection")

    database_connection = VideoMetadataDB("test.db")
    database_connection.create_table()

    sample_video = ("vid_001", "Why?", "English", True)
    database_connection.insert_video_metadata(sample_video)

    database_connection.select_all_rows()
    database_connection.close()

if __name__ == "__main__":
    main()
