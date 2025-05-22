try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' module is not installed. Please install it using 'pip install streamlit'")

import cv2
import numpy as np
import os
from PIL import Image

# Set database path
DB_PATH = r"D:\Majorproject\SOCOFing\Real"

st.title("Surface Sleuth - Fingerprint Matcher")

# File uploader for fingerprint
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    sample = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if sample is not None:
        st.image(sample, caption="Query Fingerprint", channels="GRAY")

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(sample, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        best_score = 0
        best_filename = None
        best_image = None
        best_kp2 = None
        best_matches = []

        # Search database
        for file in os.listdir(DB_PATH):
            full_path = os.path.join(DB_PATH, file)
            if not os.path.isfile(full_path):
                continue

            img = cv2.imread(full_path, 0)
            if img is None:
                continue

            kp2, des2 = orb.detectAndCompute(img, None)
            if des1 is None or des2 is None:
                continue

            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            keypoints = min(len(kp1), len(kp2))
            if keypoints == 0:
                continue
            match_percent = len(good_matches) / keypoints * 100

            if match_percent > best_score:
                best_score = match_percent
                best_filename = file
                best_image = img
                best_kp2 = kp2
                best_matches = good_matches

        if best_image is not None:
            st.success(f"Best Match: {best_filename}\nMatching Percentage: {best_score:.2f}%")

            result = cv2.drawMatches(sample, kp1, best_image, best_kp2, best_matches, None, flags=2)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Matching Result", use_container_width=True)
        else:
            st.warning("No suitable match found.")
    else:
        st.error("Error reading the uploaded image.")
