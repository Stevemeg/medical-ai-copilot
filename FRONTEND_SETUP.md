# Running the new custom frontend

This replaces the Streamlit UI (`app.py`) with a real, custom-built
HTML/CSS/JS frontend and a small FastAPI backend. `app.py` still exists
and still works independently if you ever want the Streamlit version back
-- this is an additional way to run the same underlying RAG pipeline, not
a replacement for the Python backend logic in `backend/rag_pipeline.py`
and `embeddings/retrieve.py`, which are completely unchanged.

## Why two processes now

A real, fully custom interface (true pixel control, custom animations, no
fighting a framework's default component styling) needs a real frontend
serving real HTML/CSS/JS, separate from the Python backend that does the
actual retrieval and generation work. This is the standard architecture
for this kind of app -- it's not extra complexity for its own sake.

## Step 1: Install the new dependency

```
pip install fastapi
```

(`uvicorn`, the server that runs FastAPI, was likely already installed as
a dependency of the `groq` package -- if `uvicorn --version` fails, also
run `pip install uvicorn`.)

## Step 2: Start the API backend

From the project root:

```
uvicorn api_server:app --reload --port 8000
```

Leave this running. You should see Uvicorn's startup log and
`Application startup complete.` Confirm it's working by opening
`http://localhost:8000/api/health` in a browser -- it should show
`{"status":"ok"}`.

## Step 3: Open the frontend

Just open `frontend/index.html` directly in your browser (double-click
it, or right-click → Open with → your browser). No build step, no server
needed for the frontend itself -- it's a single static HTML file that
calls the API backend via `fetch()`.

If your browser blocks `fetch()` calls from a `file://` page for security
reasons (some browsers do), serve the frontend folder with Python's
built-in server instead:

```
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in your browser.

## What to expect

- If the API backend isn't running, you'll see a red banner at the top of
  the page ("Can't reach the API server...") instead of a silent failure.
- The example question chips and "Indexed sources" panel work identically
  to before, just rendered with full custom styling instead of Streamlit
  widgets.
- Citations group by document with combined, sorted page ranges (e.g.
  "NICE NG19 — Diabetic Foot Problems · pp.6-7, 13-16, 27-29"), same logic
  verified in the Streamlit version, now living in `api_server.py`.