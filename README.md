# LLM Integration for Enhancing Technical Ticket Resolution

This project presents the integration of a large language model (LLM) into an existing technical support ticket system to assist support staff in resolving inquiries more efficiently. 
The system is built around a fine-tuned [Gemma-2-9b-it](https://ai.google.dev/gemma) model enhanced with the Retrieval-Augmented Generation (RAG) pipeline, designed to improve technical ticket resolution.
Alongside few-shot learning, prompt engineering, and semantic classification and search via SentenceTransformer, the system provides a user-friendly interface to create an assistive AI tool tailored to real-world support workflows.
The system is accessed via a Streamlit interface and optionally served through an API.


## Features

- Multilingual support
- Automated solution generation (relevant past tickets and AI-powered suggestions)
- Enhanced user experience (UX)
- Easy to use
- Written in Python


## Installation
### Prerequisites

- Python 3.10 or higher
- `pip` (Python package installer)
- Git
- Google Colab or a machine with a GPU (for inference/fine-tuning)
- Hugging Face account (optional, if using Hugging Face Hub)

### 1. Clone the repository
```bash
git clone https://github.com/srvamiri/evidenz-llm.git
cd evidenz-llm
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r app/config/requirements.txt
```
The `requirements.txt` file should include:
```
transformers
peft
accelerate
bitsandbytes
pandas
sentence-transformers
datasets
fsspec==2025.3.2
fastapi
uvicorn
faiss-cpu
streamlit
```
### 4. Run the `run.sh` file
```bash
./run.sh
```
