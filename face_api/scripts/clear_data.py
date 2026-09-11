"""
Script to clear all data from the database and image storage.
This is a destructive operation!
"""
import sys
import os

# Add project root to path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.storage import get_storage  # type: ignore
from app.core.vector_db import get_vector_db  # type: ignore

def main():
    print("WARNING: This will delete ALL data (persons, images, embeddings)!")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return

    print("\n1. Clearing SQLite database and images...")
    storage = get_storage()
    storage.reset()
    print("   Done.")

    print("\n2. Clearing Vector Database (FAISS)...")
    vector_db = get_vector_db()
    vector_db.reset()
    print("   Done.")

    print("\nAll data has been cleared successfully.")

if __name__ == "__main__":
    main()
