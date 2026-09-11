"""
Database Inspector
View contents of the SQLite database directly.
"""
import sqlite3
import os
from app.config import SQLITE_PATH, IMAGES_DIR # type: ignore

def inspect_database():
    """Print all database contents."""
    print("=" * 60)
    print("DATABASE INSPECTOR")
    print("=" * 60)
    
    print(f"\nDatabase path: {SQLITE_PATH}")
    print(f"Images directory: {IMAGES_DIR}")
    
    if not os.path.exists(SQLITE_PATH):
        print("\n⚠️ Database file does not exist yet.")
        print("Add some images first using the API!")
        return
    
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    
    # Persons
    print("\n--- PERSONS ---")
    persons = conn.execute("SELECT * FROM persons").fetchall()
    if persons:
        for p in persons:
            print(f"\nID: {p['id']}")
            print(f"  Image count: {p['image_count']}")
            print(f"  FAISS index: {p['faiss_index']}")
            print(f"  Created: {p['created_at']}")
    else:
        print("No persons in database")
    
    # Images
    print("\n--- IMAGES ---")
    images = conn.execute("SELECT * FROM images").fetchall()
    if images:
        for img in images:
            print(f"\nID: {img['id']}")
            print(f"  Person: {img['person_id']}")
            print(f"  Score: {img['score']}")
            print(f"  Path: {img['image_path']}")
            exists = "✅" if os.path.exists(img['image_path']) else "❌"
            print(f"  File exists: {exists}")
    else:
        print("No images in database")
    
    # Stats
    print(f"\n--- STATS ---")
    print(f"Total persons: {len(persons)}")
    print(f"Total images: {len(images)}")
    
    # Check FAISS index
    from app.config import FAISS_INDEX_PATH # type: ignore
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"FAISS index exists: ✅")
    else:
        print(f"FAISS index exists: ❌ (not created yet)")
    
    conn.close()


if __name__ == "__main__":
    inspect_database()
