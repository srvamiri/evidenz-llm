#!/bin/bash

# Change directory to the project root (optional if already there)
cd "$(dirname "$0")"

# Optional: Activate virtual environment if you're using one
# source venv/bin/activate

# Run the FastAPI app via Uvicorn
# chmod +x run.sh
# exec uvicorn app.main:app --host 0.0.0.0 --port 8000
# nohup ./run.sh > server.log 2>&1 &

tmux new-session -d -s mysession 'uvicorn app.main:app --host 0.0.0.0 --port 8000'
tmux split-window -h 'streamlit run app/streamlit_app.py'
tmux attach-session -t mysession