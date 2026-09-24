# Alternative HTML patient workspace

The primary deployed interface is `app.py` (Streamlit). `frontend/index.html` is a second client for the same synthetic patient workflow and Ask Evidence API. It requires the FastAPI server.

From the repository root with Python 3.11+:

```bash
pip install -r requirements.txt
uvicorn api_server:app --port 8000
```

In another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`. The HTML client calls `http://localhost:8000`; change the `API` constant in `frontend/index.html` if the server uses another address. Its tabs show Patients, Clinical Review, Ask Evidence, and Knowledge Sources. Demo fixtures and imported patient snapshots use a local SQLite file that is ignored by Git. Do not import real patient records; the prototype has no authentication or production data protections.
