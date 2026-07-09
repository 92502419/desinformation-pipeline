#!/usr/bin/env python3
"""
download_african_datasets.py — Téléchargement des datasets africains pour le pipeline
======================================================================================
Sources :
  1. MasakhaNEWS  (HuggingFace) — Vraies nouvelles en 16 langues africaines
  2. AfriSenti    (HuggingFace) — Sentiment Twitter en 14 langues africaines (news réelles)
  3. Africa Check RSS           — Faux articles vérifiés (fact-checking africain)
  4. Google News RSS africain   — Articles récents AFP/RFI/BBC Africa (vrais)

Usage :
  python scripts/download_african_datasets.py
  python scripts/download_african_datasets.py --source masakhane
  python scripts/download_african_datasets.py --source africa_check
  python scripts/download_african_datasets.py --source rss
  python scripts/download_african_datasets.py --all

Résultat : data/raw/africa_news/africa_news.csv (enrichi)
"""

import os, sys, csv, re, time, argparse, logging
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent
OUT_FILE   = BASE_DIR / 'data' / 'raw' / 'africa_news' / 'africa_news.csv'
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

HEADERS = ['title', 'body', 'label', 'source', 'language']


def clean(text: str) -> str:
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000]


def load_existing() -> list:
    """Charge les données existantes pour éviter les doublons."""
    rows = []
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        log.info(f"Données existantes : {len(rows)} exemples dans {OUT_FILE.name}")
    return rows


def save_rows(rows: list):
    with open(OUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Sauvegarde : {len(rows)} exemples → {OUT_FILE}")


# ── Source 1 : MasakhaNEWS (HuggingFace) ──────────────────────────────────────
def download_masakhane(max_per_lang: int = 200) -> list:
    """
    Télécharge MasakhaNEWS depuis HuggingFace.
    Dataset : masakhane/masakhanews — vrais articles de presse africaine.
    Labels : 0 = réel (toutes les news viennent de sources légitimes)
    Langues : amh hau ibo lin orm pcm sna som swa tir yor
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Installer datasets : pip install datasets")
        return []

    LANGS = ['amh', 'hau', 'ibo', 'lin', 'orm', 'pcm', 'sna', 'som', 'swa', 'tir', 'yor']
    LANG_MAP = {
        'amh': 'am', 'hau': 'ha', 'ibo': 'ig', 'lin': 'ln',
        'orm': 'om', 'pcm': 'pc', 'sna': 'sn', 'som': 'so',
        'swa': 'sw', 'tir': 'ti', 'yor': 'yo'
    }
    rows = []
    for lang in LANGS:
        try:
            log.info(f"  MasakhaNEWS/{lang} — chargement...")
            ds = load_dataset('masakhane/masakhanews', lang, split='train',
                              trust_remote_code=True)
            count = 0
            for item in ds:
                if count >= max_per_lang:
                    break
                title = clean(item.get('headline') or item.get('title') or '')
                body  = clean(item.get('text') or item.get('content') or '')
                if len(title) > 10:
                    rows.append({
                        'title': title,
                        'body':  body,
                        'label': 0,  # toutes les news MasakhaNEWS sont réelles
                        'source': 'masakhane',
                        'language': LANG_MAP.get(lang, lang)
                    })
                    count += 1
            log.info(f"  → {count} articles ({lang})")
        except Exception as e:
            log.warning(f"  MasakhaNEWS/{lang} : {e}")
    log.info(f"MasakhaNEWS total : {len(rows)} exemples réels")
    return rows


# ── Source 2 : Africa Check RSS (fact-checks = fake confirmés) ────────────────
AFRICA_CHECK_FEEDS = [
    ('https://africacheck.org/feed/', 'Africa Check', 'en'),
    ('https://africacheck.org/fr/feed/', 'Africa Check FR', 'fr'),
]

def download_africa_check(limit: int = 100) -> list:
    """
    Télécharge les fact-checks d'Africa Check (RSS).
    Africa Check identifie les fake news africaines → label = 1 (fake).
    """
    rows = []
    headers_http = {'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}

    for url, src_name, lang in AFRICA_CHECK_FEEDS:
        try:
            req = urllib.request.Request(url, headers=headers_http)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode('utf-8', errors='replace')

            root = ET.fromstring(content)
            count = 0
            for item in root.findall('.//item'):
                if count >= limit // len(AFRICA_CHECK_FEEDS):
                    break
                title = clean(item.findtext('title') or '')
                body  = clean(item.findtext('description') or '')
                if len(title) > 10:
                    # Africa Check titres de fact-checks = reformulation du faux article
                    # Ex: "FAUX : Le gouvernement n'a pas annoncé..."
                    rows.append({
                        'title':    title,
                        'body':     body,
                        'label':    1,  # fact-checks = fake news identifiées
                        'source':   src_name,
                        'language': lang,
                    })
                    count += 1
            log.info(f"Africa Check ({lang}) : {count} fact-checks récupérés")
            time.sleep(1)
        except Exception as e:
            log.warning(f"Africa Check ({lang}) : {e}")

    return rows


# ── Source 3 : Google News RSS — sources africaines fiables ───────────────────
AFRICAN_NEWS_QUERIES = [
    # Actualités générales africaines (sources fiables → label=0)
    ('site:rfi.fr Afrique', 'fr', 0),
    ('site:afp.com Afrique', 'fr', 0),
    ('site:bbc.com/afrique', 'fr', 0),
    ('Africa Reuters news', 'en', 0),
    ('Africa BBC news', 'en', 0),
    ('Afrique jeune actualité économique', 'fr', 0),
    ('sahel actualité sécurité AFP', 'fr', 0),
    ('West Africa news economy development', 'en', 0),
    ('East Africa Kenya Tanzania Rwanda', 'en', 0),
    ('Afrique de l\'Ouest économie développement', 'fr', 0),
    ('Southern Africa news Zimbabwe Zambia Mozambique', 'en', 0),
    ('Afrique centrale Congo Cameroun Gabon actualité', 'fr', 0),
]

def download_google_news_rss(max_per_query: int = 10) -> list:
    """Récupère des vraies nouvelles africaines via Google News RSS."""
    rows = []
    headers_http = {'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}

    for query, lang, label in AFRICAN_NEWS_QUERIES:
        q_encoded = urllib.parse.quote(query)
        hl = 'fr' if lang == 'fr' else 'en'
        gl = 'SN' if lang == 'fr' else 'NG'  # Sénégal / Nigeria
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl={hl}&gl={gl}&ceid={gl}:{hl}"
        try:
            req = urllib.request.Request(url, headers=headers_http)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode('utf-8', errors='replace')

            root = ET.fromstring(content)
            count = 0
            for item in root.findall('.//item'):
                if count >= max_per_query:
                    break
                raw_title = item.findtext('title') or ''
                raw_desc  = item.findtext('description') or ''
                source_el = item.find('source')
                src = source_el.text if source_el is not None else 'Google News Africa'
                title = clean(raw_title)
                body  = clean(raw_desc)
                if len(title) > 10:
                    rows.append({
                        'title':    title,
                        'body':     body,
                        'label':    label,
                        'source':   src,
                        'language': lang,
                    })
                    count += 1
            log.info(f"Google News RSS '{query[:30]}...' : {count} articles")
            time.sleep(2)
        except Exception as e:
            log.warning(f"Google News RSS '{query[:30]}...' : {e}")

    return rows


# ── Source 4 : Dataset synthétique étendu (fake news africaines typiques) ─────
def create_extended_synthetic() -> list:
    """
    Dataset synthétique étendu couvrant un large spectre de fake news africaines
    et d'informations réelles. Conçu pour couvrir de nombreux pays et langues.
    """
    return [
        # ─ FAKE NEWS — Santé (fr) ─────────────────────────────────────────────
        {"title": "Un chercheur camerounais prouve que les antirétroviraux causent le cancer dans 80% des cas", "body": "Un soi-disant chercheur publie sur les réseaux sociaux des données non vérifiées affirmant que les médicaments antirétroviraux distribuent par l'OMS en Afrique centrale causent le cancer.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "ALERTE : Des nanoparticules magnétiques découvertes dans les vaccins distribués au Sahel", "body": "Des vidéos non vérifiées circulent sur WhatsApp montrant prétendument des aimants adhérant aux bras de personnes vaccinées au Niger, Mali et Burkina Faso.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "Un guérisseur éthiopien affirme avoir trouvé le remède définitif contre l'hépatite B", "body": "Un homme se présentant comme guérisseur affirme détenir un extrait de plante capable de guérir l'hépatite B en trois semaines. Aucune preuve scientifique n'est fournie.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "Des milliers d'enfants rendus sourds par les vaccins méningite au Tchad selon un médecin anonyme", "body": "Un prétendu médecin anonyme affirme sur Facebook que les vaccins méningite distribués dans les régions rurales du Tchad ont rendu des milliers d'enfants sourds.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "Le sel iodé distribué en Afrique de l'Ouest contient un agent de stérilisation secret", "body": "Une publication virale sur les réseaux sociaux affirme que le sel iodé distribué dans les pays CEDEAO contiendrait une substance causant l'infertilité des femmes.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Santé (fr) ─────────────────────────────────────────────────
        {"title": "L'OMS confirme une réduction de 40% des cas de paludisme en Afrique subsaharienne", "body": "Genève (OMS) — Un rapport publié par l'Organisation mondiale de la santé indique une baisse significative des cas de paludisme en Afrique subsaharienne grâce aux nouvelles moustiquaires et au vaccin RTS,S.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "L'UNICEF distribue 2 millions de doses de vaccin contre la polio en RDC", "body": "Kinshasa (AFP) — L'UNICEF a lancé une campagne de vaccination contre la poliomyélite dans les provinces orientales de la RDC, ciblant 2 millions d'enfants de moins de 5 ans.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "Nouveau centre de recherche médicale inauguré à Lagos pour lutter contre les maladies tropicales", "body": "Lagos (AFP) — Le Nigeria a inauguré un centre de recherche médicale de pointe dédié à l'étude des maladies tropicales endémiques, avec le soutien de l'Union africaine et de l'OMS.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        # ─ FAKE NEWS — Politique africaine (fr) ──────────────────────────────
        {"title": "EXCLUSIF : Le président ivoirien a secrètement transféré 50 milliards à des paradis fiscaux", "body": "Une source anonyme non vérifiée affirme disposer de documents bancaires prouvant que des fonds publics ivoiriens ont été détournés vers des comptes offshore.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "RÉVÉLATION : L'armée française prépare une intervention militaire secrète au Tchad", "body": "Des rumeurs non confirmées circulant sur les réseaux sociaux tchadiens affirment qu'une intervention militaire française secrète est en cours de planification.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "CHOC : Le gouvernement éthiopien assassine des journalistes en secret depuis 2020", "body": "Un document non vérifié prétendu provenir de sources internes révèle l'existence d'une liste de journalistes ciblés par les services secrets éthiopiens.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "URGENT : Coup d'état militaire en cours au Sénégal — soldats dans les rues de Dakar", "body": "Des rumeurs non confirmées circulent sur WhatsApp indiquant des mouvements militaires inhabituels à Dakar. Aucune source officielle ne confirme ces informations.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "Les résultats des élections au Cameroun auraient été falsifiés par un système informatique français", "body": "Une publication virale sur Facebook affirme, sans preuves, que les résultats électoraux camerounais seraient manipulés par un logiciel développé par des ingénieurs français.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Politique africaine (fr) ───────────────────────────────────
        {"title": "L'Union africaine adopte une nouvelle charte sur la gouvernance et l'état de droit", "body": "Addis-Abeba (UA) — Les 55 États membres de l'Union africaine ont adopté une nouvelle charte sur la gouvernance démocratique et le renforcement de l'état de droit en Afrique.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "Le Sénégal organise des élections locales pacifiques avec un taux de participation de 45%", "body": "Dakar (AFP) — Les élections municipales sénégalaises se sont déroulées dans le calme selon la Commission électorale nationale autonome, avec un taux de participation de 45%.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "La CEDEAO impose des sanctions ciblées pour soutenir la transition démocratique", "body": "Abuja (CEDEAO) — La Communauté économique des États de l'Afrique de l'Ouest a adopté des sanctions ciblées pour appuyer les efforts de transition vers un gouvernement civil.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        # ─ FAKE NEWS — Économie africaine (fr) ───────────────────────────────
        {"title": "La France vole secrètement l'or africain : 60 milliards par an selon un expert", "body": "Un prétendu économiste publie un article affirmant que la France extrait chaque année 60 milliards de dollars d'or africain via le franc CFA sans compensation.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "La Banque mondiale impose délibérément l'appauvrissement des pays africains pour maintenir leur dépendance", "body": "Un texte viral sur les réseaux sociaux affirme, sans source vérifiable, que la Banque mondiale aurait une politique délibérée d'appauvrissement de l'Afrique.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "ALERTE : Les multinationales minières empoisonnent délibérément les rivières congolaises", "body": "Une vidéo non sourcée prétend montrer des multinationales versant délibérément des déchets toxiques dans les rivières proches des mines de cobalt en RDC.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Économie africaine (fr) ────────────────────────────────────
        {"title": "La croissance économique de l'Afrique subsaharienne atteint 4,1% en 2024 selon la BAD", "body": "Abidjan (BAD) — La Banque africaine de développement prévoit une croissance de 4,1% pour l'Afrique subsaharienne en 2024, tirée par les secteurs agricole et numérique.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "Le corridor commercial Abidjan-Lagos réduit les coûts du transport de 25%", "body": "Abidjan (Agence Ecofin) — La modernisation du corridor commercial Abidjan-Lagos a permis de réduire les coûts de transport de 25% pour les commerçants de la sous-région.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "L'Éthiopie rejoint la zone de libre-échange continentale africaine avec des exportations record", "body": "Addis-Abeba (Reuters) — L'Éthiopie a enregistré un volume record d'exportations dans le cadre de la Zone de libre-échange continentale africaine (ZLECAf).", "label": 0, "source": "synthetic_africa", "language": "fr"},
        # ─ FAKE NEWS — Technologie/Environnement (fr) ────────────────────────
        {"title": "Les antennes 5G installées en Afrique du Sud émettent des ondes qui causent des mutations génétiques", "body": "Une publication sur Facebook affirme que les antennes 5G récemment installées à Johannesburg causent des mutations génétiques chez les habitants proches.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "RÉVÉLÉ : Des satellites militaires déclenchent artificiellement des sécheresses en Afrique de l'Est", "body": "Un prétendu scientifique affirme sur YouTube que des satellites militaires étrangers manipulent les précipitations en Afrique de l'Est pour créer des famines artificielles.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Technologie/Environnement (fr) ─────────────────────────────
        {"title": "Le Kenya inaugure le plus grand parc éolien d'Afrique avec 365 turbines", "body": "Nairobi (AFP) — Le Kenya a inauguré le parc éolien de Lake Turkana d'une capacité de 310 MW, le plus grand parc éolien d'Afrique, fournissant de l'électricité à 150 000 foyers.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "L'Afrique du Sud déploie la première flotte de bus électriques à énergie solaire du continent", "body": "Johannesburg (Reuters) — L'Afrique du Sud a lancé une flotte de 100 bus électriques alimentés par des panneaux solaires dans le cadre d'un programme de mobilité verte.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        # ─ FAKE NEWS — English ────────────────────────────────────────────────
        {"title": "Nigerian politician caught stealing 10 billion dollars from pension fund new documents prove", "body": "Unverified documents circulating on social media purportedly show that a Nigerian senator embezzled 10 billion dollars from the national pension fund.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "SHOCKING: Kenya's President secretly signed agreement to sell Nairobi to Chinese investors", "body": "A purported leaked document shared on WhatsApp claims Kenya's president signed an agreement to cede Nairobi's city center to Chinese investors for 99 years.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "Secret lab in Uganda developing bioweapon targeting only Black Africans confirmed by whistleblower", "body": "An anonymous source claiming to be a former laboratory technician alleges that a secret laboratory in Uganda is developing a biological weapon targeting specific genetic markers.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "Ethiopian Airlines hiding plane crashes from government says insider", "body": "An unverified social media post claims an Ethiopian Airlines insider revealed that the airline has been concealing at least three plane crashes from authorities.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "BREAKING: South African army staging coup against Ramaphosa government tonight", "body": "Unverified messages spreading on WhatsApp groups claim South African military units are preparing to overthrow the elected government. No official source confirms this.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "Vaccines distributed in Ghana causing autism in children says viral post", "body": "A widely shared social media post falsely claims that vaccines recently distributed in Ghana are causing autism, citing no scientific evidence.", "label": 1, "source": "synthetic_africa", "language": "en"},
        # ─ RÉEL — English ─────────────────────────────────────────────────────
        {"title": "Kenya secures 1.8 billion dollar green energy investment from international consortium", "body": "NAIROBI (Reuters) - Kenya signed an agreement with an international investment consortium to fund 1.8 billion dollars in renewable energy projects over the next five years.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "West African nations agree on common digital identity framework at ECOWAS summit", "body": "ABUJA (AFP) - West African leaders at the ECOWAS summit approved a common digital identity framework to facilitate trade, reduce fraud and improve public service delivery.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Ghana launches landmark e-passport initiative to modernise travel documentation", "body": "ACCRA (Ghana News Agency) - Ghana launched a new biometric e-passport programme that will reduce processing times and strengthen document security across the country.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Nigeria's fintech sector records 2 billion dollars in investment in first half of year", "body": "LAGOS (Reuters) - Nigeria's financial technology sector attracted over 2 billion dollars in investment in the first six months, making it Africa's largest fintech hub.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Africa CDC reports significant progress in malaria elimination across Eastern Africa", "body": "NAIROBI (Africa CDC) - Africa CDC announced significant progress in malaria elimination across six Eastern African countries, with case rates down by an average of 32 percent.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Tanzania's Mount Kilimanjaro glacier study reveals 40 years of climate data", "body": "DAR ES SALAAM (BBC Africa) - A new scientific study on Mount Kilimanjaro's glaciers provides 40 years of detailed climate data, offering insights into East Africa's changing climate.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Senegal inaugurates West Africa's first offshore floating wind farm", "body": "DAKAR (AFP) - Senegal inaugurated the first offshore floating wind farm in West Africa off its Atlantic coast, with a capacity of 250 MW to power coastal cities.", "label": 0, "source": "synthetic_africa", "language": "en"},
        # ─ Langues africaines — Swahili ───────────────────────────────────────
        {"title": "Serikali ya Kenya itapanga fedha zaidi kwa elimu ya watoto maskini", "body": "Nairobi (AFP) — Serikali ya Kenya imetangaza mpango mpya wa kuongeza ufadhili kwa elimu ya watoto kutoka familia maskini katika maeneo ya vijijini.", "label": 0, "source": "synthetic_africa", "language": "sw"},
        {"title": "Tanzania inajenga hospitali mpya ya kisasa katika kila mkoa", "body": "Dar es Salaam (TNA) — Serikali ya Tanzania imeanza ujenzi wa hospitali za kisasa katika mikoa yote ya nchi kama sehemu ya mpango wa kuboresha huduma za afya.", "label": 0, "source": "synthetic_africa", "language": "sw"},
        {"title": "HABARI ZA UONGO: Chanjo ya COVID-19 inasababisha kifo cha haraka kwa Waafrika", "body": "Ujumbe wa uongo unaosambazwa kwa WhatsApp unasema kwamba chanjo ya COVID-19 inasababisha kifo cha haraka kwa Waafrika. Habari hizi hazina ukweli wowote.", "label": 1, "source": "synthetic_africa", "language": "sw"},
        {"title": "UONGO: Rais wa Tanzania amekamatwa kwa ufisadi wa mabilioni ya shilingi", "body": "Ujumbe usiothibitishwa unaenea mitandaoni ukidai kuwa rais amekamatwa kwa ufisadi. Hakuna chanzo rasmi kinachothibitisha habari hizi.", "label": 1, "source": "synthetic_africa", "language": "sw"},
        {"title": "Uganda yapata mafanikio makubwa katika uzalishaji wa nishati ya jua", "body": "Kampala (Reuters) — Uganda imefikia lengo lake la kuzalisha asilimia 40 ya umeme kutoka vyanzo vya nishati jadidifu, ikiwa ni mafanikio makubwa ya nchi.", "label": 0, "source": "synthetic_africa", "language": "sw"},
        # ─ Langues africaines — Hausa ─────────────────────────────────────────
        {"title": "Gwamnatin Najeriya ta kaddamar da shirin tallafawa manoma don bunkasa noma", "body": "Abuja (AFP) — Gwamnatin Najeriya ta kaddamar da shiri na musamman don taimakawa manoma su samu kayan noma da kudin aiki a farashi mai rahusa.", "label": 0, "source": "synthetic_africa", "language": "ha"},
        {"title": "Cutar ta sirri ta kashe dubban mutane a Najeriya amma gwamnati tana boyewa", "body": "Wani sakon da ba a tabbatar da sahihancinsa ba na yadi akan WhatsApp yana cewa an kashe dubban mutane da cutar ta asiri a Najeriya, kuma gwamnati na boye labarin.", "label": 1, "source": "synthetic_africa", "language": "ha"},
        {"title": "Kasar Ghana ta sami lambar yabo ta kasashen duniya don ci gaban tattalin arziki", "body": "Accra (Reuters) — Ghana ta sami lambar yabo ta kasashen duniya don ci gaba mai dorewa a harkar tattalin arziki, musamman a bangaren fasahar sadarwa.", "label": 0, "source": "synthetic_africa", "language": "ha"},
        # ─ Langues africaines — Yoruba ────────────────────────────────────────
        {"title": "Ijoba Naijiria ti kede eto tuntun fun imudara ẹkọ awọn ọmọde ni agbegbe igberiko", "body": "Abuja (AFP) — Ijoba apapọ Naijiria ti kede eto tuntun kan lati mu ẹkọ dara si fun awọn ọmọde ni awọn agbegbe igberiko pẹlu iye owo naira bilionu marun.", "label": 0, "source": "synthetic_africa", "language": "yo"},
        {"title": "IFO: Ajesara COVID-19 n fa aarun ara fun awọn ọmọ Africa", "body": "Ifiranṣẹ iro ti n tan kaakiri lori WhatsApp sọ pe ajesara COVID-19 n fa awọn aisan nla fun awọn ọmọ Africa. Eyi jẹ iro patapata.", "label": 1, "source": "synthetic_africa", "language": "yo"},
        # ─ Langues africaines — Wolof ─────────────────────────────────────────
        {"title": "Senegaal am na kib bi ci energi soleer yëngu beneen rekk ci Afrik Initerew", "body": "Dakar (AFP) — Senegaal daaneel ci suñu mbedd bi ak yëp ci kanam bu metti ji, ci energi soleer bi, war na di yëgël mbëgël.", "label": 0, "source": "synthetic_africa", "language": "wo"},
        {"title": "DEF CI XEETU GARAB: Garab bu lañu def ci Senegaal moo def waññi yaram", "body": "Xabaar bu dëgërul yi lëndëm ci WhatsApp dafa wax ne garab yu toppatoo yi lañu jox ci Senegaal dafa doon daf yaram yu meen. Loolu dëgg du.", "label": 1, "source": "synthetic_africa", "language": "wo"},
        # ─ Langues africaines — Amharique ─────────────────────────────────────
        {"title": "ኢትዮጵያ የፀሃይ ኃይል ፕሮጀክቶቿን አስፋፍታለች", "body": "አዲስ አበባ (AFP) — ኢትዮጵያ ታዳሽ ኃይልን ለማሳደግ 500 ሚሊዮን ዶላር ወጪ የሚያስፈልጋቸው አዳዲስ የፀሃይ ኃይል ፕሮጀክቶችን ማስጀመሯን ታወጀ።", "label": 0, "source": "synthetic_africa", "language": "am"},
        {"title": "ሐሰተኛ ዜና: ክትባቱ ሕዝቡን ለመቆጣጠር የሚያገለግል ቺፕ አለው", "body": "ሐሰተኛ ዜና በዋትሳፕ ላይ እየሰራጨ ሲሆን ክትባቱ ሰዎችን ለመቆጣጠር የሚያስችለው ማይክሮ ቺፕ እንደያዘ ይናገራል። ይህ ሙሉ በሙሉ የሐሰት ዜና ነው።", "label": 1, "source": "synthetic_africa", "language": "am"},
        # ─ Langues africaines — Lingala ───────────────────────────────────────
        {"title": "RDC esali eloko ya monene na domaine ya nishati ya lisungama", "body": "Kinshasa (AFP) — République Démocratique du Congo esali eloko ya monene na domaine ya nishati ya lisungama, na ko-initier projet ya GW 2 ya courant electrique.", "label": 0, "source": "synthetic_africa", "language": "ln"},
        {"title": "LOKUTA: Gouvernement ya RDC ebundeli bato mpe ebomba yango", "body": "Message ya lokuta ezali ko-tanda na WhatsApp ezali koloba ete gouvernement ya RDC ebundeli bato mpe ebomba yango. Makanisi wana ezali lokuta.", "label": 1, "source": "synthetic_africa", "language": "ln"},
        # ─ Fake news — Complotisme global/Afrique (fr) ───────────────────────
        {"title": "Bill Gates finance secrètement la stérilisation forcée des femmes en Afrique subsaharienne", "body": "Une publication virale sur les réseaux sociaux accuse Bill Gates de financer un programme secret de stérilisation forcée ciblant les femmes africaines via des ONG de santé.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "La CIA finance les groupes djihadistes au Sahel pour déstabiliser les gouvernements pro-russes", "body": "Un texte non sourcé prétend que la CIA finance secrètement les groupes terroristes au Sahel pour renverser les gouvernements qui se rapprochent de la Russie.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "RÉVÉLATION : Les accords de Cotonou servent à piller les ressources africaines légalement", "body": "Un économiste dissident publie un essai non vérifié affirmant que les accords de partenariat économique UE-Afrique sont un mécanisme légal de pillage des ressources.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Afrique du Nord (fr/ar) ────────────────────────────────────
        {"title": "Le Maroc renforce son réseau ferroviaire à grande vitesse avec 3 nouvelles lignes", "body": "Rabat (MAP) — Le Maroc a lancé les travaux de 3 nouvelles lignes de train à grande vitesse reliant les principales villes du royaume, avec un investissement de 22 milliards de dirhams.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "La Tunisie lance un plan d'urgence pour relancer le tourisme après des années de crise", "body": "Tunis (TAP) — Le gouvernement tunisien a annoncé un plan d'urgence de relance du secteur touristique, visant à attirer 8 millions de visiteurs étrangers cette année.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "L'Algérie investit 7 milliards de dollars dans la transition vers l'énergie verte d'ici 2030", "body": "Alger (APS) — L'Algérie a annoncé un plan d'investissement de 7 milliards de dollars dans les énergies renouvelables pour atteindre 15 GW d'énergie propre d'ici 2030.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        # ─ FAKE — Afrique du Nord (fr) ────────────────────────────────────────
        {"title": "EXCLUSIF : Le gouvernement marocain torture des dissidents dans des prisons secrètes révèle Amnesty", "body": "Une publication non vérifiée sur les réseaux sociaux se présentant comme un rapport d'Amnesty International dénonce l'existence de prisons secrètes au Maroc.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "ALERTE : Des soldats tunisiens préparent un coup d'état pour renverser le président", "body": "Des messages non confirmés circulent sur WhatsApp signalant des mouvements militaires à Tunis. Aucune source officielle ne confirme ces informations.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ RÉEL — Afrique Australe (en) ──────────────────────────────────────
        {"title": "Zambia achieves full repayment of IMF emergency credit facility", "body": "LUSAKA (Reuters) - Zambia completed the repayment of its IMF emergency credit facility ahead of schedule, signalling a return to macroeconomic stability after years of debt distress.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Zimbabwe launches new agricultural recovery plan targeting 2 million smallholder farmers", "body": "HARARE (AFP) - Zimbabwe launched a comprehensive agricultural recovery plan targeting 2 million smallholder farmers with inputs, training and market access support.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Botswana and Namibia sign joint water management agreement for shared river basin", "body": "GABORONE (AFP) - Botswana and Namibia signed a landmark water management agreement to jointly manage the Okavango River Basin shared between the two countries.", "label": 0, "source": "synthetic_africa", "language": "en"},
        # ─ FAKE — Afrique Australe (en) ──────────────────────────────────────
        {"title": "CONFIRMED: South Africa plans to expel all white farmers and seize land without compensation by 2025", "body": "A misleading social media post falsely claims that South Africa has announced a plan to expel white farmers and confiscate all farm land without compensation.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "Zimbabwe secret gold reserves worth 50 trillion stolen by Western banks revealed by whistleblower", "body": "An unverified claim circulating online alleges Zimbabwe possesses 50 trillion dollars in hidden gold reserves that have been secretly stolen by Western financial institutions.", "label": 1, "source": "synthetic_africa", "language": "en"},
        # ─ Couverture Sahel (fr) ─────────────────────────────────────────────
        {"title": "Le Mali renforce son système éducatif avec l'aide de la Russie et de la Turquie", "body": "Bamako (AFP) — Le Mali a signé des accords de coopération éducative avec la Russie et la Turquie pour former des enseignants et équiper les écoles rurales.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "Le Niger relance la production agricole grâce à des projets d'irrigation innovants", "body": "Niamey (RFI) — Le Niger a lancé 15 projets d'irrigation solaire qui permettent aux agriculteurs de cultiver pendant la saison sèche, réduisant l'insécurité alimentaire.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "Le Burkina Faso reçoit des équipements médicaux d'urgence du Comité international de la Croix-Rouge", "body": "Ouagadougou (CICR) — Le Comité international de la Croix-Rouge a livré du matériel médical d'urgence aux hôpitaux de campagne dans les zones affectées par les crises sécuritaires.", "label": 0, "source": "synthetic_africa", "language": "fr"},
        {"title": "MENSONGE : Des soldats maliens massacrent des civils en secret sur ordre du gouvernement russe", "body": "Une publication non vérifiée sur les réseaux sociaux accuse les forces armées maliennes et les instructeurs russes de massacres civils. Ces informations ne sont pas confirmées.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        {"title": "FAUX : Le gouvernement nigérien a vendu ses réserves d'uranium à l'Iran en secret", "body": "Un message viral sur Facebook affirme que le gouvernement nigérien a signé un accord secret pour vendre ses réserves d'uranium à l'Iran. Aucune preuve n'existe.", "label": 1, "source": "synthetic_africa", "language": "fr"},
        # ─ Afrique de l'Est — anglais ─────────────────────────────────────────
        {"title": "Rwanda launches ambitious 5-year digital transformation masterplan", "body": "KIGALI (Rwanda Development Board) - Rwanda launched a comprehensive digital transformation masterplan targeting a 90 percent digital economy by 2030, focusing on fintech, smart cities and e-health.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Uganda and Kenya complete cross-border railway link boosting regional trade", "body": "NAIROBI (Reuters) - Uganda and Kenya officially inaugurated the cross-border railway link between Kampala and Nairobi, expected to reduce freight costs by 40 percent.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "Ethiopia's largest ever dam begins commercial power generation", "body": "ADDIS ABABA (AFP) - Ethiopia's Grand Ethiopian Renaissance Dam began commercial electricity generation, providing an estimated 6,000 MW that will transform energy access across the region.", "label": 0, "source": "synthetic_africa", "language": "en"},
        {"title": "FAKE: Uganda's Museveni arrested by own military according to viral WhatsApp message", "body": "A false WhatsApp message claiming Uganda's president has been arrested by his own military is circulating. Ugandan authorities confirm this is completely false.", "label": 1, "source": "synthetic_africa", "language": "en"},
        {"title": "FALSE: Rwanda secretly testing mind-control substances in new national vaccine program", "body": "A fabricated social media post falsely claims Rwanda is testing mind-control substances through its national vaccine program. This claim is entirely false.", "label": 1, "source": "synthetic_africa", "language": "en"},
    ]


# ── Fonction principale ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Téléchargement des datasets africains')
    parser.add_argument('--source', choices=['masakhane', 'africa_check', 'rss', 'synthetic', 'all'],
                        default='all')
    parser.add_argument('--max-per-lang', type=int, default=200,
                        help='Nombre max d\'articles par langue pour MasakhaNEWS')
    args = parser.parse_args()

    existing = load_existing()
    existing_titles = {r.get('title', '').strip().lower() for r in existing}

    new_rows = []

    if args.source in ('synthetic', 'all'):
        log.info("Génération des exemples synthétiques étendus...")
        syn_rows = create_extended_synthetic()
        added = 0
        for r in syn_rows:
            if r['title'].strip().lower() not in existing_titles:
                new_rows.append(r)
                existing_titles.add(r['title'].strip().lower())
                added += 1
        log.info(f"  Synthétiques : +{added} nouveaux exemples")

    if args.source in ('masakhane', 'all'):
        log.info("Téléchargement MasakhaNEWS (HuggingFace)...")
        msk_rows = download_masakhane(max_per_lang=args.max_per_lang)
        added = 0
        for r in msk_rows:
            if r['title'].strip().lower() not in existing_titles:
                new_rows.append(r)
                existing_titles.add(r['title'].strip().lower())
                added += 1
        log.info(f"  MasakhaNEWS : +{added} nouveaux exemples")

    if args.source in ('africa_check', 'all'):
        log.info("Téléchargement Africa Check RSS (fact-checks)...")
        ac_rows = download_africa_check(limit=200)
        added = 0
        for r in ac_rows:
            if r['title'].strip().lower() not in existing_titles:
                new_rows.append(r)
                existing_titles.add(r['title'].strip().lower())
                added += 1
        log.info(f"  Africa Check : +{added} nouveaux exemples")

    if args.source in ('rss', 'all'):
        log.info("Téléchargement Google News RSS (sources africaines)...")
        rss_rows = download_google_news_rss(max_per_query=10)
        added = 0
        for r in rss_rows:
            if r['title'].strip().lower() not in existing_titles:
                new_rows.append(r)
                existing_titles.add(r['title'].strip().lower())
                added += 1
        log.info(f"  Google News RSS : +{added} nouveaux exemples")

    all_rows = existing + new_rows
    fake_count = sum(1 for r in all_rows if str(r.get('label', '')) == '1')
    real_count = sum(1 for r in all_rows if str(r.get('label', '')) == '0')

    log.info(f"\nTotal final : {len(all_rows)} exemples "
             f"({fake_count} fake / {real_count} réel)")

    save_rows(all_rows)
    log.info("✅ Dataset africain mis à jour.")
    log.info("   Prochaine étape : python scripts/preprocess_data.py && python scripts/train_model.py")


if __name__ == '__main__':
    main()
