#!/usr/bin/env python3
"""
Database migration script to add source column to health_records table
"""

import sqlite3
import os
from database import DATABASE_URL

def migrate_database():
    """Add source column to existing health_records table"""
    
    # Extract database path from URL
    db_path = DATABASE_URL.replace("sqlite:///./", "")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if source column already exists
        cursor.execute("PRAGMA table_info(health_records)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'source' in columns:
            print("Source column already exists")
            conn.close()
            return True
        
        # Add source column with default value
        cursor.execute("""
            ALTER TABLE health_records 
            ADD COLUMN source TEXT DEFAULT 'manual'
        """)
        
        # Update existing records to have 'manual' source
        cursor.execute("""
            UPDATE health_records 
            SET source = 'manual' 
            WHERE source IS NULL
        """)
        
        conn.commit()
        conn.close()
        
        print("Successfully added source column to health_records table")
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("Database Migration: Adding source column")
    print("=" * 50)
    
    success = migrate_database()
    
    if success:
        print("\nMigration completed successfully!")
        print("You can now restart the server.")
    else:
        print("\nMigration failed. Check the error above.")