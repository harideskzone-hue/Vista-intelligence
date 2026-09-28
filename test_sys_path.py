import sys
import os

print("Sys path:", sys.path)
try:
    from app.core.storage import Storage
    print("Successfully imported Storage via app.core.storage")
except Exception as e:
    print("Failed app import:", e)

try:
    from face_api.app.core.storage import Storage
    print("Successfully imported Storage via face_api.app.core.storage")
except Exception as e:
    print("Failed face_api import:", e)
