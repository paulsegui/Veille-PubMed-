import os
import re
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
    '("radiomics"[Title/Abstract] AND "embolization"[Title/Abstract]) OR '
    '"Frandon"[Author] OR '
    '"Sapoval"[Author] OR '
    '"Bommart"[Author] OR '
    '"Guiu"[Author] OR '
    '"CERIMED"[Affiliation] OR '
    '"LIIE"[Affiliation]'
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

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

# Nombre d'articles envoyes par appel IA (pour rester sous la limite de tokens/minute du compte gratuit)
BATCH_SIZE = 6
# Delai de base entre deux lots, pour laisser le quota TPM se recharger
BASE_DELAY_BETWEEN_BATCHES = 25


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


def parse_retry_after(error_text):
    """Extrait le delai suggere par Groq dans le message d'erreur (ex: 'try again in 21.285s')."""
    match = re.search(r"try again in ([\d.]+)s", error_text)
    if match:
        return float(match.group(1)) + 2  # petite marge de securite
    return None


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
            wait = parse_retry_after(r.text) or (15 * (attempt + 1))
            time.sleep(wait)
            continue
        break
    return None


# Contexte personnel : decrit qui je suis et mes projets de recherche actuels,
# pour que l'IA priorise selon ma pratique reelle et pas seulement des mots-cles.
PROFILE_CONTEXT = (
    "CONTEXTE ME CONCERNANT (a utiliser pour juger la pertinence, en plus des criteres ci-dessous) :\n\n"
    "Je suis medecin junior (docteur junior) en radiologie interventionnelle "
    "vasculaire au CHU de Montpellier, en parcours hospitalo-universitaire. "
    "Mes projets de recherche actuels sont :\n"
    "- Radiomique IRM appliquee a la prediction de reponse a l'embolisation "
    "de prostate (PAE) : tout article sur la radiomique/IRM predictive en "
    "embolisation ou en uro-radiologie interventionnelle m'interesse "
    "particulierement.\n"
    "- Developpement d'un agent d'embolisation particulaire bio-source a base "
    "d'agar-agar (projet Embobio, dans le cadre d'un Master 2 et d'un stage de "
    "recherche a Marseille/CERIMED) : tout article sur les nouveaux agents "
    "emboliques, biomateriaux, particules bio-sourcees ou alternatives aux "
    "particules calibrees classiques est tres pertinent.\n"
    "- Manuscrits en cours sur le syndrome de FAVA et les malformations "
    "vasculaires a bas debit (registre BAMARA), notamment la sclerotherapie "
    "et la dissociation IRM-clinique post-traitement : tout article sur ces "
    "sujets est tres pertinent.\n"
    "- Publication anterieure sur la courbe d'apprentissage du stenting "
    "iliofemoral : les articles sur le stenting veineux, la recanalisation "
    "veineuse et le syndrome post-thrombotique m'interessent directement.\n\n"
    "Utilise ce contexte pour hausser la priorite des articles qui recoupent "
    "precisement ces projets (meme s'ils semblent nicher), et pour degrader "
    "la priorite d'articles qui ne font que mentionner des mots-cles generaux "
    "sans lien avec ma pratique reelle.\n\n"
)

PROMPT_INSTRUCTIONS = (
    "Tu es mon assistant personnel de veille scientifique.\n\n"
    "Je suis radiologue interventionnel avec un interet particulier\n"
    "pour la radiologie interventionnelle vasculaire et les techniques\n"
    "d'embolisation.\n\n"
    + PROFILE_CONTEXT +
    "Ta mission est de filtrer les nouveaux articles scientifiques\n"
    "ci-dessous.\n\n"
    "NE RETIENS QUE les articles reellement utiles ou importants. Les\n"
    "articles hors-sujet ou sans rapport avec ma pratique ne doivent meme\n"
    "pas etre mentionnes (ni titre, ni raison de rejet) : ignore-les\n"
    "completement et silencieusement.\n\n"
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
    "IMPORTANT - niveau de detail selon la categorie :\n\n"
    "Pour les articles classes \U0001F525 A LIRE uniquement, donne le detail\n"
    "complet :\n"
    "- pourquoi il est important (2-3 phrases)\n"
    "- question etudiee, population, methode, resultats principaux\n"
    "- limites importantes\n"
    "- ce que cela pourrait changer en pratique\n"
    "- si cela peut inspirer une etude / publication\n\n"
    "Pour les articles classes \U0001F7E0 INTERESSANT, donne SEULEMENT 2 a 3\n"
    "phrases maximum : le resultat principal et pourquoi ca vaut le coup\n"
    "d'y jeter un oeil, rien de plus (pas de section population/methode\n"
    "detaillee).\n\n"
    "Pour les articles classes \U0001F4A1 IDEE DE RECHERCHE, donne SEULEMENT\n"
    "1 a 2 phrases : le lien avec une idee de recherche possible, sans\n"
    "detailler l'etude elle-meme.\n\n"
    "Classe les articles en :\n\n"
    "\U0001F525 A LIRE\n"
    "\U0001F7E0 INTERESSANT\n"
    "\U0001F4A1 IDEE DE RECHERCHE\n\n"
    "Si aucun article n'est reellement important dans ce lot, ecris juste\n"
    "'Rien de notable dans ce lot.' sans autre commentaire.\n\n"
    "NE FABRIQUE AUCUNE information absente de l'abstract.\n\n"
    "Utilise uniquement les informations fournies.\n\n"
)

CONSOLIDATION_INSTRUCTIONS = (
    "Voici plusieurs analyses partielles d'articles scientifiques, produites\n"
    "lot par lot (le meme prompt de filtrage a ete applique a chaque lot).\n"
    "Fusionne-les en UN SEUL document final au format HTML, propre et non\n"
    "redondant.\n\n"
    "FORMAT DE SORTIE OBLIGATOIRE :\n"
    "- Reponds UNIQUEMENT avec du HTML (pas de markdown, pas de ```html,\n"
    "  pas de texte avant/apres) ; ne mets PAS de balises <html>, <head> ou\n"
    "  <body>, seulement le contenu interne (des <h2>, <h3>, <p>, <a>, etc.)\n"
    "- Titre de section pour les articles a lire : "
    "<h2 style=\"color:#c0392b;border-bottom:2px solid #c0392b;"
    "padding-bottom:4px;\">\U0001F525 A LIRE</h2>\n"
    "- Titre de section pour les articles interessants : "
    "<h2 style=\"color:#e67e22;border-bottom:2px solid #e67e22;"
    "padding-bottom:4px;\">\U0001F7E0 INTERESSANT</h2>\n"
    "- Titre de section pour les idees de recherche : "
    "<h2 style=\"color:#2980b9;border-bottom:2px solid #2980b9;"
    "padding-bottom:4px;\">\U0001F4A1 IDEE DE RECHERCHE</h2>\n"
    "- N'affiche une section que si elle contient au moins un article\n"
    "- Pour chaque article : <h3 style=\"margin-bottom:2px;\">Titre de "
    "l'article</h3>, puis <p style=\"margin-top:2px;color:#555;font-style:"
    "italic;\">Nom du journal</p>, puis le texte d'analyse dans un ou "
    "plusieurs <p>, puis <p><a href=\"LIEN_PUBMED\" style=\"color:#2980b9;\">"
    "Voir sur PubMed</a></p>\n"
    "- Separe chaque article par <hr style=\"border:none;border-top:1px "
    "solid #ddd;margin:16px 0;\">\n\n"
    "REGLES DE CONTENU :\n"
    "- Regroupe tous les articles \U0001F525 A LIRE ensemble en premier "
    "(avec le detail complet, sans le raccourcir davantage)\n"
    "- Puis tous les \U0001F7E0 INTERESSANT ensemble (garde le format court)\n"
    "- Puis tous les \U0001F4A1 IDEE DE RECHERCHE ensemble (garde le format "
    "tres court)\n"
    "- Supprime toute phrase du type 'rien de notable dans ce lot' ou toute\n"
    "  reference aux lots eux-memes : le lecteur ne doit jamais savoir que\n"
    "  le traitement a ete fait par petits paquets\n"
    "- Ne reformule pas le contenu deja ecrit, contente-toi de reorganiser, "
    "nettoyer et mettre en forme en HTML\n"
    "- Si absolument aucun article n'a ete retenu dans aucun lot, reponds "
    "juste avec <p>Aucun article notable dans cette periode.</p>\n\n"
    "Analyses partielles a fusionner :\n\n"
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


def consolidate(batch_results):
    joined = "\n\n===LOT SUIVANT===\n\n".join(batch_results)
    prompt = CONSOLIDATION_INSTRUCTIONS + joined

    for model in GROQ_MODELS:
        result = call_groq(model, prompt)
        if result:
            return result
    # Si la consolidation echoue, on renvoie les lots bruts plutot que rien
    return joined


def summarize_with_ai(articles):
    if not articles:
        return "<p>Aucun nouvel article trouve dans la periode selectionnee.</p>", True

    batches = list(chunked(articles, BATCH_SIZE))
    results = []
    any_success = False

    for idx, batch in enumerate(batches):
        result = summarize_batch(batch)
        if result:
            any_success = True
            results.append(result)
        else:
            raw = "\n\n".join(
                f"- {a['title']} ({a['journal']})\n  {a['url']}" for a in batch
            )
            results.append(f"[Resume IA indisponible pour ce lot, articles bruts]\n\n{raw}")
        if idx < len(batches) - 1:
            time.sleep(BASE_DELAY_BETWEEN_BATCHES)

    if not any_success:
        items = "".join(
            f'<li><b>{a["title"]}</b> ({a["journal"]}) - '
            f'<a href="{a["url"]}">Voir sur PubMed</a></li>'
            for a in articles
        )
        html = (
            "<p>Le resume automatique n'a pas pu etre genere "
            "(services IA indisponibles). Voici les articles bruts trouves :</p>"
            f"<ul>{items}</ul>"
        )
        return html, False

    time.sleep(BASE_DELAY_BETWEEN_BATCHES)
    final = consolidate(results)

    return final, any_success


def send_email(summary_html, nb_articles, ai_ok):
    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    date_str = datetime.utcnow().strftime("%d/%m/%Y")
    tag = "" if ai_ok else " [resume brut]"
    msg["Subject"] = f"Veille radiologie interventionnelle - {nb_articles} article(s) - {date_str}{tag}"

    full_html = f"""\
<html>
<body style="font-family: Arial, Helvetica, sans-serif; max-width: 700px;
             margin: 0 auto; color: #222; line-height: 1.5;">
  <h1 style="font-size: 20px; color: #333; border-bottom: 3px solid #333;
             padding-bottom: 8px;">
    Veille radiologie interventionnelle vasculaire - {date_str}
  </h1>
  <p style="color:#777; font-size: 13px;">{nb_articles} article(s) trouve(s) sur la periode.</p>
  {summary_html}
</body>
</html>
"""

    msg.attach(MIMEText(full_html, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


if __name__ == "__main__":
    pmids = search_pubmed()
    articles = fetch_details(pmids)
    summary, ai_ok = summarize_with_ai(articles)
    send_email(summary, len(articles), ai_ok)
    print(f"Termine. {len(articles)} article(s) traite(s). IA utilisee: {ai_ok}")
