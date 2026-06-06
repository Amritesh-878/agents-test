# 👋 START HERE — Run the Class Transcription

Hi! This program listens to the recorded classes and writes down what everyone said. Your
computer's graphics card (GPU) makes it fast and accurate. **Just follow the steps in order
and copy-paste the commands exactly.** It takes a while but most of it is just waiting.

> If you ever see **red error text**, don't worry — nothing is broken. Take a photo of the
> whole window and send it back, and you'll get told what to do.

---

## Part 1 — One-time setup (~30 min, mostly waiting)

### Step 1 — Install Python 3.11
1. Go to **https://www.python.org/downloads/release/python-3119/**
2. Scroll to the bottom, click **"Windows installer (64-bit)"**, and run the downloaded file.
3. **VERY IMPORTANT:** on the first screen, tick the box **"Add python.exe to PATH"** at the bottom.
4. Click **"Install Now"**, wait, then **Close**.

### Step 2 — Install ffmpeg (lets it read the audio)
1. Click **Start**, type **PowerShell**, **right-click** "Windows PowerShell" → **Run as administrator**.
2. Paste this, press **Enter**:
   ```
   winget install Gyan.FFmpeg
   ```
3. If it asks you to agree, type **Y** and Enter. Wait until it finishes, then **close that window**.

### Step 3 — Open the project folder as a command window
1. Unzip the folder I sent you (right-click → **Extract All**).
2. Open the unzipped folder so you can see files like `requirements.txt` and a folder called `Videos`.
3. Click the **address bar** at the top of the window (where the folder path is), type **powershell**, press **Enter**. A dark window opens — it's already in the right place.

### Step 4 — Install the program (run these one at a time, wait for each to finish)
Paste each line, press Enter, and **wait until it stops** before the next one:
```
py -3.11 -m venv .venv
```
```
.\.venv\Scripts\Activate.ps1
```
> If that line shows a red **"running scripts is disabled"** error, paste this once and try it again:
> ```
> Set-ExecutionPolicy -Scope Process -Bypass
> ```
```
python -m pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
```
python -m pip install -r requirements.txt
```
The last one downloads a lot and can take 10–20 minutes. Lots of text scrolling is normal.

---

## Part 2 — Run it (the main part)

### Step 5 — Stop the computer from sleeping
Settings → System → Power → set **"Screen and sleep"** to **Never** (while plugged in). This run
takes hours and the computer must stay awake. Keep it **plugged in**.

### Step 6 — Start the transcription
Paste this **one** command and press Enter:
```
python -m scripts.run_pipeline --input Videos --output-dir output --teacher "Nisha" --model medium --skip-embed
```
- The **first time**, it downloads the AI model (~1.5 GB) — needs internet, just wait.
- Then it processes every class. This takes **several hours** (maybe overnight). Leave it running.
- You'll see lots of text scroll by — that's normal and good.
- When it's finished it prints something like **"Pipeline complete: 9/9 classes succeeded"**.

> If the window closed or the PC restarted: re-open PowerShell in the folder (Step 3), run
> `.\.venv\Scripts\Activate.ps1` first, then the Step 6 command again — it skips work already done.

---

## Part 3 — Send it back

1. Open the folder named **`output`**. Inside, each class has its own folder, and inside *those*
   is a folder called **`raw`**. **Delete every `raw` folder** — they're just copies of the videos
   I already have, and they make the zip huge. Everything else stays.
   > Quick way: paste this in the same PowerShell window to delete them all at once:
   > ```
   > Get-ChildItem output -Recurse -Directory -Filter raw | Remove-Item -Recurse -Force
   > ```
2. Go back up, **right-click** the **`output`** folder → **Send to** → **Compressed (zipped) folder**.
3. Send me that **`output.zip`**. 🎉 That's it — thank you so much!!

---

## If something goes wrong
- **Any red error** → photo of the whole window, send it to me. Easy to fix, not your fault.
- **"out of memory" / "CUDA"** error → tell me; you'll get a smaller setting to use.
- **"ffmpeg not found"** → close PowerShell, re-do Step 3 to open a fresh one, try again.
- **Stuck for more than a few minutes with no new text** → that's usually normal (it's thinking); give it time. If unsure, send a photo.
