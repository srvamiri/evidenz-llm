from app.modules import AutoModelForCausalLM, AutoTokenizer, PeftModel, BitsAndBytesConfig, torch
from app.hugging_face_config import load_hf

load_hf()
model_path = "google/gemma-2-9b-it"
model_repo = "samiri1377/gemma-2-9b-it-fine-tuned"

_model = None
_tokenizer = None

# Define quantization config (choose 8-bit or 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",
    quantization_config=bnb_config,
    device_map="auto",
)

def load_model():
    try:
        global _model, _tokenizer
        if _model is None:
            # Load the tokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            # Load the fine-tuned model
            _model = PeftModel.from_pretrained(base_model, model_repo)

        return _model, _tokenizer
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)
