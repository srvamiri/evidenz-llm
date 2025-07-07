from app.modules import login as hf_login
from app.modules import os, torch

logged_in = False

def hugging_face_login():
    global logged_in
    if os.path.exists("app/config/hugging_face_token.txt"):
        with open("app/config/hugging_face_token.txt", "r") as f:
            hf_token = f.read().strip()
        hf_login(hf_token)
        logged_in = True
    else:
        raise FileNotFoundError("Hugging Face token file not found. Please create 'config/hugging_face_token.txt' with your token.")

def load_hf():
    try:
        if not logged_in:
            hugging_face_login()
            torch.set_default_device("cuda")        
    except FileNotFoundError:
        print("❌ Hugging Face Login failed.")
        