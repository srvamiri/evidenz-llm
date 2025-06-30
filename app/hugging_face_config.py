from modules import login as hf_login
from modules import os, torch, HfApi, HfHubHTTPError

hf_api = HfApi()

def hugging_face_login():
    if os.path.exists("app/config/hugging_face_token.txt"):
        with open("app/config/hugging_face_token.txt", "r") as f:
            hf_token = f.read().strip()
        hf_login(hf_token)
        return True
    else:
        raise FileNotFoundError("Hugging Face token file not found. Please create 'app/config/hugging_face_token.txt' with your token.")

def load_hf():
    try:
        user_info = hf_api.whoami()
        if user_info:
            return True
        else:
            hugging_face_login()
            torch.set_default_device("cuda")
    except HfHubHTTPError:
        print("❌ Hugging Face Login failed.")