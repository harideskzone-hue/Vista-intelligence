import cv2
try:
    cv2.putText = lambda *args, **kwargs: None
    print("PATCH WORKED")
except Exception as e:
    print("PATCH FAILED:", type(e), e)
