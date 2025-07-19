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
### ✅ Prerequisites

- Python 3.10 or higher
- `pip  install app/requirements.txt`
- Git
- Google Colab or a machine with GPU (for inference/fine-tuning)
- Hugging Face account (optional, if using Hugging Face Hub)

