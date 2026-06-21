# CarID Deployment Guide — Hugging Face Spaces

## Step 1 — Export your FAISS index from Lightning AI

Run this in your Lightning AI terminal to create a zip of everything needed:

```bash
cd /teamspace/studios/this_studio

# Create a deployment package
mkdir -p carid_deploy
cp main.py rag.py fallback_handler.py db_updater.py car_id_chat_app.html carid_deploy/
cp data/faiss.index data/metadata_flat.json carid_deploy/

# Zip it
zip -r carid_deploy.zip carid_deploy/
echo "Done — download carid_deploy.zip"
```

## Step 2 — Create the Hugging Face Space

1. Go to **huggingface.co/new-space**
2. Fill in:
   - **Space name:** `carid-backend` (or any name)
   - **SDK:** Docker  ← important
   - **Visibility:** Public (free) or Private
3. Click **Create Space**

## Step 3 — Upload your files

In your new Space, upload these files (drag & drop or use the web editor):

```
Dockerfile
requirements.txt
README.md
main.py
rag.py
fallback_handler.py
db_updater.py
car_id_chat_app.html
faiss.index          ← from data/faiss.index on Lightning AI
metadata_flat.json   ← from data/metadata_flat.json on Lightning AI
```

**Important:** `faiss.index` and `metadata_flat.json` go in the ROOT of the Space
(not in a `data/` folder), because the Dockerfile sets WORKDIR to /app.

Update `main.py` paths for HF Spaces — the data files will be at `/app/`:

```python
BASE          = Path("/app")           # changed from /teamspace/studios/...
IMAGES_DIR    = Path("/app/images")    # images not needed after index is built
METADATA_FILE = Path(os.getenv("METADATA_FILE", str(BASE / "metadata_flat.json")))
INDEX_FILE    = Path(os.getenv("INDEX_FILE",    str(BASE / "faiss.index")))
```

Also change the HTML serving path in `serve_frontend()`:
```python
for candidate in [
    Path("/app/car_id_chat_app.html"),
    Path("car_id_chat_app.html"),
]:
```

## Step 4 — Add secret API keys

In your Space → **Settings** → **Repository secrets**, add:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `SERPER_API_KEY` | your Serper key |

Do NOT put keys in the code or README.

## Step 5 — Build and test

After uploading all files, HF Spaces will automatically build the Docker image.
Build takes 3–8 minutes. Watch the build logs in the **Logs** tab.

When build finishes, your app is live at:
```
https://YOUR_USERNAME-carid-backend.hf.space
```

Test it:
```bash
curl https://YOUR_USERNAME-carid-backend.hf.space/health
```

## Step 6 — Keep it awake (free trick)

HF Spaces sleep after 48h of inactivity. Use UptimeRobot to ping it:

1. Go to **uptimerobot.com** → free account
2. Add monitor: HTTP → `https://YOUR_USERNAME-carid-backend.hf.space/health`
3. Interval: every 30 minutes
4. Your Space stays awake 24/7 for free

## Share with your team

Give your team this URL:
```
https://YOUR_USERNAME-carid-backend.hf.space
```

They open it in any browser — no login, no setup needed.
