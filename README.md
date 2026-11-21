# ResumeLens: AI Resume Screening Assistant

Ranks candidates against a job description **by meaning**, using a pretrained sentence Transformer
(all-MiniLM-L6-v2), and shows the evidence behind every score.

## Folder
```
resumelens/
├── app.py          # Streamlit interface
├── matcher.py      # ALL NLP logic (read this to understand the project)
├── skills.py       # skill dictionary
├── samples.py      # demo job description + 4 fictional candidates
├── test_core.py    # self-check, run it tonight
├── eval_data.py    # 3 jobs x 5 labelled resumes (for the report)
├── evaluate.py     # keyword vs Transformer vs hybrid: Top-1 and Spearman
├── demo_files/     # job_description.txt + 5 resume PDFs (for the live upload demo)
├── requirements.txt
```

## Install and run (Python 3.9+, no GPU)
```
cd resumelens
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python test_core.py
python evaluate.py             
streamlit run app.py
```
* **First run needs internet** to download the model (about 90 MB). Do this TONIGHT. Afterwards it works offline.
* If the model cannot load, the app does NOT crash: it shows a red banner and uses a keyword (TF-IDF) fallback.
  Do not demo in fallback mode; fix the download first.
* Offline backup copy:
  `python -c "from huggingface_hub import snapshot_download as s; s('sentence-transformers/all-MiniLM-L6-v2', local_dir='minilm-local')"`
  then run with `RESUMELENS_MODEL=minilm-local` (Windows: `set RESUMELENS_MODEL=minilm-local`).

## Using it
1. Demo data is loaded by default. Look at the **Ranking dashboard**.
2. **Candidate deep-dive**: evidence sentence for every requirement.
3. **Transformer vs keywords**: type two sentences and compare both methods.
4. Switch the sidebar to **Upload my own** and upload `demo_files/job_description.txt` (JD box) and PDFs from `demo_files/`
   (try `Meera_Iyer.pdf`, which is not in the built-in demo).
5. Turn on **Blind screening**.

## IMPORTANT before execution
Run `python test_core.py` and read the printed ranking. The thresholds (0.50 strong / 0.35 partial) are sensible
defaults for this model but were NOT calibrated on the real weights. If something looks off, adjust the sliders in
the sidebar and decide which values you will use, then write them in your report.
