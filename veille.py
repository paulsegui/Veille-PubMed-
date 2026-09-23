import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests
import time

PUBMED_QUERY = os.environ.get(
    "PUBMED_QUERY",
    '("interventional radiology"[Title/Abstract] OR "embolization"[Title/Abstract] '
    'OR "endovascular"[Title/Abstract] OR "vascular malformation"[Title/Abstract] '
    'OR "chemoembolization"[Title/Abstract] OR "prostatic artery embolization"[Title/Abstract])'
)
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", GMAIL_ADDRESS)
DAYS_BACK = int(os.environ.get("DAYS_BACK", "3"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

def search_pubmed():
    date_max = datetime.utcnow().strftime("%Y/%m/%d")
    date_min = (datetime.utcnow() - timedelta(days=DAYS_BACK)).strftime("%Y/%m/%d")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": PUBMED_QUERY,
        "mindate": date_min,
        "maxdate": date_max,
        "datetype": "pdat",
        "retmax": 30,
        "retmode": "json",
        "sort": "most+recent",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_details(pmids):
    if not pmids:
        return []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = art.findtext(".//ArticleTitle") or "(sans titre)"
        abstract = " ".join(
            (a.text or "") for a in art.findall(".//AbstractText")
        ) or "(pas de resume disponible)"
        journal = art.findtext(".//Journal/Title") or ""
        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


def summarize_with_gemini(articles):
    if not articles:
        return "Aucun nouvel article trouve dans la periode selectionnee."

    corpus = "\n\n".join(
        f"Titre: {a['title']}\nJournal: {a['journal']}\nResume: {a['abstract']}\nLien: {a['url']}"
        for a in articles
    )

    prompt = (
        "Tu es un assistant qui aide un medecin en radiologie interventionnelle "
        "vasculaire a faire sa veille scientifique. Voici une liste d'articles "
        "PubMed recents. Pour chaque article pertinent en radiologie "
        "interventionnelle vasculaire, redige en francais : le titre, un resume "
        "de 3-4 phrases de l'apport clinique ou scientifique principal, et le "
        "lien PubMed. Ignore les articles hors-sujet ou de tres faible interet "
        "clinique. Classe du plus au moins important.\n\n"
        f"Articles :\n{corpus}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    last_error = None
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code in (503, 429):
                wait = 15 * (attempt + 1)
                print(f"Serveur surcharge (code {r.status_code}), nouvelle tentative dans {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(10)

    raise RuntimeError(f"Echec apres plusieurs tentatives: {last_error}")


def send_email(summary, nb_articles):
    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    date_str = datetime.utcnow().strftime("%d/%m/%Y")
    msg["Subject"] = f"Veille radiologie interventionnelle - {nb_articles} article(s) - {date_str}"
    msg.attach(MIMEText(summary, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


if __name__ == "__main__":
    pmids = search_pubmed()
    articles = fetch_details(pmids)
    summary = summarize_with_gemini(articles)
    send_email(summary, len(articles))
    print(f"Termine. {len(articles)} article(s) traite(s).")
