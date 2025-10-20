from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import instaloader
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("model.h5")

def scrape_instagram_features(username: str):
    L = instaloader.Instaloader(download_pictures=False, download_videos=False, quiet=True)
    profile = instaloader.Profile.from_username(L.context, username)

    has_profile_pic = 1 if profile.profile_pic_url else 0
    username_nums_ratio = sum(c.isdigit() for c in profile.username) / len(profile.username)
    fullname_words = len(profile.full_name.split()) if profile.full_name else 0
    fullname_nums_ratio = (
        sum(c.isdigit() for c in profile.full_name) / len(profile.full_name)
        if profile.full_name else 0
    )
    name_equals_username = int(profile.full_name.lower() == profile.username.lower())
    description_length = len(profile.biography) if profile.biography else 0
    has_external_url = 1 if profile.external_url else 0
    is_private = int(profile.is_private)
    posts = profile.mediacount or 0
    followers = profile.followers or 0
    follows = profile.followees or 0

    features = {
        "profile pic": has_profile_pic,
        "nums/length username": username_nums_ratio,
        "fullname words": fullname_words,
        "nums/length fullname": fullname_nums_ratio,
        "name==username": name_equals_username,
        "description length": description_length,
        "external URL": has_external_url,
        "private": is_private,
        "#posts": posts,
        "#followers": followers,
        "#follows": follows,
    }
    return features


@app.get("/analyze/{username}")
def analyze(username: str):
    try:
        features = scrape_instagram_features(username)
        X = pd.DataFrame([features])
        X = np.array(X, dtype=np.float32)
        prob = float(model.predict(X)[0][0])
        return {"probability": prob}
    except Exception as e:
        return {"error": str(e)}