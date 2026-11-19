# Put the project online (GitHub -> Hugging Face -> Railway)

Order: A) GitHub  B) upload models  C) Railway.  Do not skip B: best.pt files cannot go on GitHub (100 MB limit).

## A. Push the project to GitHub
1. Install Git: https://git-scm.com/download/win (keep all defaults). Close and reopen VS Code.
2. Make a free account at https://github.com. Click **+ > New repository**. Name: `car-damage-app`. Choose **Private**.
   Do NOT tick "Add a README". Click Create.
3. Put the new files (Dockerfile, fetch_models.py, requirements-railway.txt, .dockerignore, .gitignore, updated main.py and
   price_scraper.py) in your project folder.
4. In VS Code: Terminal > New Terminal. Run one line at a time:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   git init
   git add .
   git status
   ```
   Read the `git status` list. It must NOT show .pt files, car_images, .venv or zip files. If it does, add them to `.gitignore`,
   then run `git rm -r --cached .` and `git add .` again.
5. ```
   git commit -m "first version"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/car-damage-app.git
   git push -u origin main
   ```
   A browser window asks you to sign in to GitHub. Approve it. Refresh the repo page: your files are there.
   Later changes: `git add .`, `git commit -m "what I changed"`, `git push`.

## B. Upload the model files to Hugging Face (private)
1. Free account at https://huggingface.co. **New Model**, name `car-damage-models`, visibility **Private**.
2. **Files > Add file > Upload**: upload the part detector best.pt as `part_best.pt`, the damage best.pt as `damage_best.pt`
   (and optionally best_26k.pt as `best_26k.pt`).
3. Settings > Access Tokens > New token, type **Read**. Copy it (starts with `hf_`). Keep it secret.
4. Your links look like `https://huggingface.co/YOUR-NAME/car-damage-models/resolve/main/part_best.pt`.

## C. Deploy on Railway
1. https://railway.com, sign in with GitHub. Choose the **Hobby** plan (about $5/month). The free plan has too little memory
   for YOLO. Source: https://docs.railway.com/pricing/plans
2. **New Project > Deploy from GitHub repo** > pick `car-damage-app` (allow Railway to see it).
   Railway finds the Dockerfile and builds. The first build takes several minutes.
3. Open the service > **Variables** and add:
   | Name | Value |
   |---|---|
   | APP_USER | surveyor |
   | APP_PASSWORD | a long password you invent |
   | PART_MODEL_URL | link of part_best.pt |
   | DAMAGE_MODEL_URL | link of damage_best.pt |
   | DAMAGE_MODEL_2_URL | (optional) link of best_26k.pt |
   | MODEL_DOWNLOAD_TOKEN | your hf_ token |
   | MAX_UPLOAD_MB | 100 |
4. Railway redeploys. Open **Deployments > View logs**. You should see "done" for the model downloads, then
   "Uvicorn running".
5. **Settings > Networking > Generate Domain**. Open that link, type the user and password. That link works from anywhere.
6. Settings > **Usage / spending limit**: set a limit so a surprise bill is impossible.

## Things to know
- Not tested by me: the real Docker build, Railway itself, and PakWheels price lookups from a cloud server (PakWheels may block
  cloud IPs; prices may fail online while damage detection works).
- CPU only: one image takes seconds, a long video can take minutes and may time out. Keep videos short.
- The Dockerfile sets DAMAGE_TTA=0 and DAMAGE_TILES=0 for speed. Add DAMAGE_TTA=1 / DAMAGE_TILES=2 in Variables for more
  sensitivity (slower).
- Uploaded files are lost whenever Railway redeploys. Download results you need.
- Photos of customers' cars go to a server: keep the password private.
