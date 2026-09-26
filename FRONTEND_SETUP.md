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

Open `http://localhost:5500`. The HTML client calls the Compose API at `http://localhost:18000`; change the `API` constant in `frontend/index.html` if the server uses another address. Its tabs show Patients, Clinical Review, Ask Evidence, and Knowledge Sources. Demo fixtures import into PostgreSQL. The HTML client is intended for explicit development mode; a production UI must obtain and send an OIDC access token. Do not import real patient records into this prototype.
