from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from decimal import Decimal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np
import torch
import pandas as pd
import faiss
import json
from peft import PeftModel
import os
import streamlit as st
import requests