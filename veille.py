import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import requests


PUBMED_QUERY = os.environ.get(
    "PUBMED_QUERY",
    '('
    '"Embolization, Therapeutic"[Mesh] OR '
    '"prostatic artery embolization"[Title/Abstract] OR '
    '"chemoembolization"[Title/Abstract] OR '
    '"transarterial embolization"[Title/Abstract] OR '
    '"vascular malformation"[Title/Abstract] OR '
    '"arteriovenous malformation"[Title/Abstract] OR '
    '"sclerotherapy"[Title/Abstract] OR '
    '"embolic agent"[Title/Abstract] OR '
    '"venous stenting"[Title/Abstract] OR '
    '"iliofemoral stenting"[Title/Abstract] OR '
    '"uterine fibroid embolization"[Title/Abstract] OR '
    '"uterine artery embolization"[Title/Abstract] OR '
    '"pelvic congestion syndrome"[Title/Abstract] OR '
    '"pelvic vein embolization"[Title/Abstract] OR '
    '"ovarian vein embolization"[Title/Abstract] OR '
    '"cardiac MRI"[Title/Abstract] OR '
    '"cardiac magnetic resonance"[Title/Abstract] OR '
    '"chest CT"[Title/Abstract] OR '
    '"thoracic CT"[Title/Abstract] OR '
    '"agar"[Title/Abstract] OR '
    '"bio-sourced embolic"[Title/Abstract] OR '
    '"biosourced embolic"[Title/Abstract] OR '
    '("radiomics"[Title/Abstract] AND "embolization"[Title/Abstract])'
    ') NOT ('
    '"aortic aneurysm"[Title/Abstract] OR '
    '"aortic dissection"[Title/Abstract] OR '
    '"carotid endarterectomy"[Title/Abstract] OR '
    '"cardiac surgery"[Title/Abstract] OR '
    '"neurointervention"[Title/Abstract] OR '
    '"neuroradiology"[Title/Abstract] OR '
    '"intracranial aneurysm"[Title/Abstract] OR '
    '"stroke thrombectomy"[Title/Abstract] OR '
    '"bypass graft"[Title/Abstract]'
    ')'
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", GMAIL_ADDRESS)
DAYS_BACK = int(os.environ.get("DAYS_BACK", "3"))

# Plusieurs modeles essayes dans l'ordre : si le premier est sature, on tente le suivant
GEMINI_MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


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


def call_gemini(model, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(3):
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"[{model}] tentative {attempt+1} echouee, code {r.status_code}: {r.text[:300]}")
        if r.status_code in (503, 429):
            time.sleep(20 * (attempt + 1))
            continue
        break  # erreur non transitoire (401, 400...), inutile d'insister sur ce modele
    return None


def summarize_with_gemini(articles):
    if not articles:
        return "Aucun nouvel article trouve dans la periode selectionnee.", True

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

    for model in GEMINI_MODELS:
        result = call_gemini(model, prompt)
        if result:
            return result, True

    # Tous les modeles ont echoue : on renvoie quand meme les articles bruts
    fallback = (
        "Le resume automatique n'a pas pu etre genere (services IA indisponibles). "
        "Voici les articles bruts trouves :\n\n"
    )
    fallback += "\n\n".join(
        f"- {a['title']} ({a['journal']})\n  {a['url']}"
        for a in articles
    )
    return fallback, False


def send_email(summary, nb_articles, ai_ok):
    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    date_str = datetime.utcnow().strftime("%d/%m/%Y")
    tag = "" if ai_ok else " [resume brut]"
    msg["Subject"] = f"Veille radiologie interventionnelle - {nb_articles} article(s) - {date_str}{tag}"
    msg.attach(MIMEText(summary, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


if __name__ == "__main__":
    pmids = search_pubmed()
    articles = fetch_details(pmids)
    summary, ai_ok = summarize_with_gemini(articles)
    send_email(summary, len(articles), ai_ok)
    print(f"Termine. {len(articles)} article(s) traite(s). IA utilisee: {ai_ok}")
    
