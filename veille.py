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
    '"embolic particle"[Title/Abstract] OR '
    '"venous stenting"[Title/Abstract] OR '
    '"iliofemoral stenting"[Title/Abstract] OR '
    '"venous recanalization"[Title/Abstract] OR '
    '"post-thrombotic syndrome"[Title/Abstract] OR '
    '"deep vein thrombosis"[Title/Abstract] OR '
    '"pulmonary embolism"[Title/Abstract] OR '
    '"uterine fibroid embolization"[Title/Abstract] OR '
    '"uterine artery embolization"[Title/Abstract] OR '
    '"pelvic congestion syndrome"[Title/Abstract] OR '
    '"pelvic vein embolization"[Title/Abstract] OR '
    '"ovarian vein embolization"[Title/Abstract] OR '
    '"bio-sourced embolic"[Title/Abstract] OR '
    '"biosourced embolic"[Title/Abstract] OR '
    '("cardiac MRI"[Title/Abstract] AND ("vascular"[Title/Abstract] OR "embolization"[Title/Abstract] OR "radiomics"[Title/Abstract])) OR '
    '("chest CT"[Title/Abstract] AND ("embolization"[Title/Abstract] OR "vascular"[Title/Abstract] OR "pulmonary embolism"[Title/Abstract])) OR '
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
    ') NOT ('
    '"Case Reports"[Publication Type] OR '
    '"Comment"[Publication Type] OR '
    '"Letter"[Publication Type] OR '
    '"Editorial"[Publication Type] OR '
    '"News"[Publication Type]'
    ')'
)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", GMAIL_ADDRESS)
DAYS_BACK = int(os.environ.get("DAYS_BACK", "3"))

# Modeles Groq essayes dans l'ordre (tous gratuits, infra separee de Google)
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

# Nombre d'articles envoyes par appel IA (pour rester sous la limite de tokens/minute du compte gratuit)
BATCH_SIZE = 6


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


def call_groq(model, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    for attempt in range(3):
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        print(f"[{model}] tentative {attempt+1} echouee, code {r.status_code}: {r.text[:300]}")
        if r.status_code in (503, 429):
            time.sleep(15 * (attempt + 1))
            continue
        break
    return None


PROMPT_INSTRUCTIONS = (
    "Tu es mon assistant personnel de veille scientifique.\n\n"
    "Je suis radiologue interventionnel avec un interet particulier\n"
    "pour la radiologie interventionnelle vasculaire et les techniques\n"
    "d'embolisation.\n\n"
    "Ta mission est de filtrer les nouveaux articles scientifiques\n"
    "ci-dessous.\n\n"
    "NE RETIENS QUE les articles reellement utiles ou importants.\n\n"
    "Priorite aux :\n\n"
    "1. essais randomises\n"
    "2. etudes prospectives importantes\n"
    "3. grandes series multicentriques\n"
    "4. meta-analyses\n"
    "5. recommandations / guidelines\n"
    "6. innovations techniques importantes\n"
    "7. comparaisons de techniques\n"
    "8. resultats pouvant modifier la pratique\n"
    "9. articles particulierement pertinents pour la RI vasculaire\n"
    "10. articles pouvant generer une idee de recherche\n\n"
    "Sois particulierement attentif a :\n\n"
    "- embolisation\n"
    "- agents emboliques\n"
    "- particules\n"
    "- coils\n"
    "- plugs\n"
    "- PAE\n"
    "- UAE\n"
    "- hemorragie\n"
    "- thrombectomie\n"
    "- DVT\n"
    "- embolie pulmonaire\n"
    "- maladie veineuse\n"
    "- syndrome post-thrombotique\n"
    "- stents veineux\n"
    "- recanalisation\n"
    "- sharp recanalization\n\n"
    "Ne selectionne PAS simplement un article parce qu'il contient\n"
    "les mots \"interventional radiology\".\n\n"
    "Pour chaque article retenu :\n\n"
    "- explique en 2-3 phrases pourquoi il est important\n"
    "- resume la question etudiee\n"
    "- donne la population\n"
    "- donne la methode\n"
    "- donne les resultats principaux\n"
    "- donne les limites importantes\n"
    "- explique ce que cela pourrait changer en pratique\n"
    "- indique si cela peut inspirer une etude / publication\n\n"
    "Classe les articles en :\n\n"
    "\U0001F525 A LIRE\n"
    "\U0001F7E0 INTERESSANT\n"
    "\U0001F4A1 IDEE DE RECHERCHE\n\n"
    "Si aucun article n'est reellement important, dis-le clairement.\n\n"
    "NE FABRIQUE AUCUNE information absente de l'abstract.\n\n"
    "Utilise uniquement les informations fournies.\n\n"
)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def summarize_batch(batch):
    corpus = "\n\n".join(
        f"Titre: {a['title']}\nJournal: {a['journal']}\nResume: {a['abstract']}\nLien: {a['url']}"
        for a in batch
    )
    prompt = PROMPT_INSTRUCTIONS + f"Articles :\n{corpus}"

    for model in GROQ_MODELS:
        result = call_groq(model, prompt)
        if result:
            return result
    return None


def summarize_with_ai(articles):
    if not articles:
        return "Aucun nouvel article trouve dans la periode selectionnee.", True

    batches = list(chunked(articles, BATCH_SIZE))
    results = []
    any_success = False

    for idx, batch in enumerate(batches):
        result = summarize_batch(batch)
        if result:
            any_success = True
            results.append(result)
        else:
            # Secours pour ce lot uniquement : articles bruts de ce lot
            raw = "\n\n".join(
                f"- {a['title']} ({a['journal']})\n  {a['url']}" for a in batch
            )
            results.append(f"[Resume IA indisponible pour ce lot, articles bruts]\n\n{raw}")
        if idx < len(batches) - 1:
            time.sleep(3)  # respecte la limite de requetes/minute entre les lots

    summary = "\n\n---\n\n".join(results)
    return summary, any_success


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
    summary, ai_ok = summarize_with_ai(articles)
    send_email(summary, len(articles), ai_ok)
    print(f"Termine. {len(articles)} article(s) traite(s). IA utilisee: {ai_ok}")
            
