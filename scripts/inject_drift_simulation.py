"""
inject_drift_simulation.py — Simulation de Concept Drift pour la soutenance
===========================================================================
Auteur  : KOMOSSI Sosso — Master BIG DATA IA, UCAO UUT 2025-2026
Usage   : python scripts/inject_drift_simulation.py [--scenario A|B|C|D] [--daemon] [--interval 900] [--initial-delay 90]

Scénarios :
  A — Dérive abrupte  : injection d'un bloc soudain d'articles très fortement fake
  B — Dérive graduelle: augmentation progressive du taux fake sur 120 messages (défaut)
  C — Dérive cyclique : alternance fake/réel pour simuler une campagne récurrente
  D — Dérive incrémentale: changement lent et continu de la distribution textuelle

Modes :
  (défaut)  : exécution unique du scénario choisi
  --daemon  : boucle continue — rejoue le scénario toutes les --interval secondes
              (utilisé par le service Docker drift-injector)

Résultats visibles dans Grafana (http://localhost:3000) et Streamlit (http://localhost:8501).
"""

import json, hashlib, time, argparse, logging, random, os
from datetime import datetime, timezone
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Depuis l'hôte : localhost:9092  |  Depuis Docker : kafka:29092
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
TOPIC        = os.getenv('KAFKA_TOPIC_RAW', 'raw-news-stream')

# ── Corpus d'articles simulés ─────────────────────────────────────────────────

FAKE_TITLES = [
    "URGENT: Le gouvernement cache une cure miracle contre le cancer découverte en Afrique",
    "RÉVÉLATION: Les vaccins COVID contiennent des nanopuces de surveillance activées par 5G",
    "EXCLUSIF: Un chercheur prouve que la Terre est plate — les preuves supprimées par l'ONU",
    "CHOC: Les élections truquées par un algorithme secret — témoignage d'un initié",
    "SCANDALE: L'eau du robinet empoisonnée délibérément pour réduire la population mondiale",
    "INCROYABLE: Des extraterrestres dirigent en secret plusieurs gouvernements africains",
    "COMPLOT: Les médias traditionnels payés pour cacher la vérité sur les crises économiques",
    "ALERTE: Un virus mutant 100x plus mortel que COVID préparé dans un laboratoire militaire",
    "RÉVÉLÉ: Les banques centrales planifient l'effondrement mondial programmé pour 2026",
    "PREUVE: Les élections présidentielles manipulées depuis 30 ans grâce à des logiciels espions",
    "STUPÉFIANT: La CIA finance secrètement les groupes terroristes au Sahel depuis 2015",
    "CONFIRMÉ: Le réchauffement climatique est une invention des multinationales pétrolières",
    "URGENT: Les réseaux sociaux censurent massivement les opposants politiques africains",
    "RÉVÉLATION: Une ONG internationale distribue des stérilisants déguisés en médicaments",
    "CHOC: Des milliers de morts cachés par le gouvernement dans les hôpitaux militaires",
]

FAKE_BODIES = [
    "Des sources anonymes très proches du pouvoir révèlent que cette information, soigneusement dissimulée, prouve l'existence d'un plan mondial visant à contrôler les populations.",
    "Un document classifié obtenu exclusivement confirme ce que beaucoup soupçonnaient : les autorités mentent depuis des années sur ce sujet crucial pour l'avenir de l'humanité.",
    "Les preuves accumulées par des chercheurs indépendants montrent sans ambiguïté que les institutions officielles cachent délibérément la vérité au grand public.",
    "Cette information explosive, partagée des millions de fois, révèle l'ampleur du complot organisé au plus haut niveau de l'État contre les citoyens ordinaires.",
    "Des témoins oculaires confirment que les faits rapportés par les médias officiels sont totalement faux et que la réalité est bien plus troublante que ce qu'on nous dit.",
]

REAL_TITLES = [
    "L'Union Africaine tient son sommet annuel à Addis-Abeba sur le développement durable",
    "Les marchés financiers mondiaux enregistrent une hausse modérée après les données d'emploi",
    "La COP30 fixe de nouveaux objectifs de réduction des émissions de carbone pour 2030",
    "Le Togo renforce sa coopération économique avec ses voisins de la CEDEAO",
    "Les chercheurs publient une étude sur les effets du changement climatique en Afrique de l'Ouest",
    "L'OMS recommande une nouvelle campagne de vaccination contre la méningite au Sahel",
    "Le FMI révise à la hausse ses prévisions de croissance pour l'Afrique subsaharienne",
    "Une conférence internationale sur la cybersécurité se tient à Lomé cette semaine",
    "Les Nations Unies appellent à un cessez-le-feu immédiat dans les zones de conflit actives",
    "De nouvelles infrastructures routières améliorent la connectivité dans plusieurs pays africains",
]

REAL_BODIES = [
    "Les dirigeants ont discuté des stratégies de développement économique inclusif lors de cette réunion.",
    "Les experts économiques analysent les tendances du marché et prévoient une stabilisation progressive.",
    "Les gouvernements s'engagent à respecter les accords internationaux sur le climat et le développement.",
    "La coopération régionale progresse avec de nouveaux accords commerciaux signés cette semaine.",
    "Les scientifiques présentent leurs conclusions après deux années de recherches approfondies sur le terrain.",
]

SOURCES_FAKE = ["InfoBrûlante", "VéritéCachée", "AlerteComplot", "RévélationsExclusives", "InfoResistance"]
SOURCES_REAL = ["AFP", "Reuters", "BBC News", "Le Monde", "RFI", "France24", "AP News"]

def make_article(is_fake: bool, idx: int, gdelt_tone: float = 0.0) -> dict:
    if is_fake:
        title = random.choice(FAKE_TITLES)
        body  = random.choice(FAKE_BODIES)
        source = random.choice(SOURCES_FAKE)
        cat = "suspicious"
    else:
        title = random.choice(REAL_TITLES)
        body  = random.choice(REAL_BODIES)
        source = random.choice(SOURCES_REAL)
        cat = "reliable"

    url = f"https://simulation.test/{source.lower().replace(' ','-')}/article-{idx}"
    art_id = hashlib.md5((url + title).encode()).hexdigest()
    return {
        "id":              art_id,
        "title":           title,
        "body":            body,
        "url":             url,
        "source":          source,
        "source_category": cat,
        "language":        "fr",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "gdelt_tone":      gdelt_tone,
    }


def send_batch(producer: Producer, articles: list):
    for art in articles:
        producer.produce(
            topic=TOPIC,
            key=art["id"],
            value=json.dumps(art, ensure_ascii=False).encode("utf-8"),
        )
    producer.flush()


# ── Scénarios ─────────────────────────────────────────────────────────────────

def scenario_a_abrupt(producer: Producer):
    """Dérive abrupte : passage brutal de ~50% fake à ~90% fake en un seul bloc."""
    log.info("=== Scénario A — Dérive Abrupte ===")
    log.info("Phase 1/2 : baseline 30 articles (50% fake)")
    send_batch(producer, [make_article(i % 2 == 0, i) for i in range(30)])
    time.sleep(6)

    log.info("Phase 2/2 : DÉRIVE — 80 articles (90% fake) — visible dans Grafana ~2 min")
    articles = [make_article(random.random() < 0.90, i + 30, gdelt_tone=-5.0)
                for i in range(80)]
    send_batch(producer, articles)
    log.info("Scénario A terminé. Ouvrez Grafana → Score Composite Drift")


def scenario_b_gradual(producer: Producer):
    """Dérive graduelle : taux fake augmente progressivement de 50% à 90% sur 120 messages."""
    log.info("=== Scénario B — Dérive Graduelle (scénario soutenance) ===")
    total = 120
    for i in range(total):
        fake_prob = 0.50 + (0.40 * i / total)   # 50% → 90% progressivement
        article = make_article(random.random() < fake_prob, i, gdelt_tone=-i * 0.05)
        producer.produce(
            topic=TOPIC,
            key=article["id"],
            value=json.dumps(article, ensure_ascii=False).encode("utf-8"),
        )
        if (i + 1) % 10 == 0:
            producer.flush()
            pct = int(fake_prob * 100)
            log.info(f"  {i+1}/{total} articles envoyés — taux fake simulé : {pct}%")
            time.sleep(2)
    producer.flush()
    log.info("Scénario B terminé. Vérifiez le Score Composite Drift dans Grafana.")


def scenario_c_cyclic(producer: Producer):
    """Dérive cyclique : 3 pics fake/réel pour simuler une campagne récurrente."""
    log.info("=== Scénario C — Dérive Cyclique ===")
    for cycle in range(3):
        log.info(f"Cycle {cycle+1}/3 — pic FAKE (30 articles 85% fake)")
        send_batch(producer, [make_article(random.random() < 0.85, cycle * 60 + i)
                               for i in range(30)])
        time.sleep(4)
        log.info(f"Cycle {cycle+1}/3 — retour RÉEL (30 articles 20% fake)")
        send_batch(producer, [make_article(random.random() < 0.20, cycle * 60 + 30 + i)
                               for i in range(30)])
        time.sleep(4)
    log.info("Scénario C terminé. Observez les pics répétés dans la timeline Grafana.")


def scenario_d_incremental(producer: Producer):
    """Dérive incrémentale lente : changement très progressif sur 200 messages."""
    log.info("=== Scénario D — Dérive Incrémentale ===")
    total = 200
    for i in range(total):
        fake_prob = 0.30 + (0.60 * (i / total) ** 2)  # courbe quadratique 30% → 90%
        article = make_article(random.random() < fake_prob, i)
        producer.produce(
            topic=TOPIC,
            key=article["id"],
            value=json.dumps(article, ensure_ascii=False).encode("utf-8"),
        )
        if (i + 1) % 20 == 0:
            producer.flush()
            log.info(f"  {i+1}/{total} articles — taux fake simulé : {fake_prob*100:.0f}%")
            time.sleep(1)
    producer.flush()
    log.info("Scénario D terminé. KSWIN est le plus adapté à cette dérive graduelle.")


# ── Fonction principale d'injection (réutilisée par l'API FastAPI) ───────────

def run_scenario(scenario: str, broker: str = None, topic: str = None) -> dict:
    """Exécute un scénario de dérive et retourne un résumé. Utilisé par l'API REST."""
    kb = broker or KAFKA_BROKER
    tp = topic  or TOPIC
    producer = Producer({
        "bootstrap.servers": kb,
        "client.id": "drift-injector",
        "message.max.bytes": 2000000,
        "linger.ms": 100,
    })

    def _send(articles):
        for art in articles:
            producer.produce(
                topic=tp,
                key=art["id"],
                value=json.dumps(art, ensure_ascii=False).encode("utf-8"),
            )
        producer.flush()

    sc = scenario.upper()
    if sc == "A":
        log.info("=== Scénario A — Dérive Abrupte ===")
        log.info("Phase 1/2 : baseline 30 articles (50% fake)")
        _send([make_article(i % 2 == 0, i) for i in range(30)])
        time.sleep(6)
        log.info("Phase 2/2 : DÉRIVE — 80 articles (90% fake)")
        _send([make_article(random.random() < 0.90, i + 30, gdelt_tone=-5.0) for i in range(80)])
    elif sc == "C":
        log.info("=== Scénario C — Dérive Cyclique ===")
        for cycle in range(3):
            log.info(f"Cycle {cycle+1}/3 — pic FAKE")
            _send([make_article(random.random() < 0.85, cycle*60+i) for i in range(30)])
            time.sleep(4)
            log.info(f"Cycle {cycle+1}/3 — retour RÉEL")
            _send([make_article(random.random() < 0.20, cycle*60+30+i) for i in range(30)])
            time.sleep(4)
    elif sc == "D":
        log.info("=== Scénario D — Dérive Incrémentale ===")
        total = 200
        for i in range(total):
            fake_prob = 0.30 + (0.60 * (i / total) ** 2)
            art = make_article(random.random() < fake_prob, i)
            producer.produce(tp, key=art["id"],
                             value=json.dumps(art, ensure_ascii=False).encode("utf-8"))
            if (i + 1) % 20 == 0:
                producer.flush()
                log.info(f"  {i+1}/{total} — {fake_prob*100:.0f}%")
                time.sleep(1)
        producer.flush()
    else:  # B (défaut)
        log.info("=== Scénario B — Dérive Graduelle ===")
        total = 120
        for i in range(total):
            fake_prob = 0.50 + (0.40 * i / total)
            art = make_article(random.random() < fake_prob, i, gdelt_tone=-i*0.05)
            producer.produce(tp, key=art["id"],
                             value=json.dumps(art, ensure_ascii=False).encode("utf-8"))
            if (i + 1) % 10 == 0:
                producer.flush()
                log.info(f"  {i+1}/{total} — {int(fake_prob*100)}%")
                time.sleep(2)
        producer.flush()

    log.info(f"✅ Scénario {sc} terminé.")
    return {"scenario": sc, "status": "injected", "broker": kb, "topic": tp}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Injecte une simulation de Concept Drift dans le pipeline."
    )
    parser.add_argument(
        "--scenario", choices=["A", "B", "C", "D"], default="B",
        help="Scénario : A=abrupt, B=graduel (défaut), C=cyclique, D=incrémental"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Mode continu : rejoue le scénario toutes les --interval secondes"
    )
    parser.add_argument(
        "--interval", type=int, default=900,
        help="Intervalle entre deux injections en mode daemon (défaut : 900s = 15 min)"
    )
    parser.add_argument(
        "--initial-delay", type=int, default=90, dest="initial_delay",
        help="Délai initial avant la première injection (défaut : 90s — laisse Spark démarrer)"
    )
    args = parser.parse_args()

    log.info("="*60)
    log.info("  SIMULATION DE CONCEPT DRIFT — PIPELINE DÉSINFORMATION")
    log.info(f"  Broker Kafka  : {KAFKA_BROKER}")
    log.info(f"  Scénario      : {args.scenario}")
    log.info(f"  Mode          : {'DAEMON (boucle)' if args.daemon else 'UNIQUE'}")
    if args.daemon:
        log.info(f"  Délai initial : {args.initial_delay}s")
        log.info(f"  Intervalle    : {args.interval}s ({args.interval//60} min)")
    log.info("="*60)

    if args.initial_delay > 0:
        log.info(f"⏳ Attente de {args.initial_delay}s pour laisser le pipeline démarrer...")
        time.sleep(args.initial_delay)

    cycle = 0
    while True:
        cycle += 1
        if args.daemon:
            log.info(f"--- Cycle {cycle} ---")
        result = run_scenario(args.scenario)
        log.info(f"✅ Injection terminée : {result}")
        log.info("   → Grafana   : http://localhost:3000")
        log.info("   → Streamlit : http://localhost:8501")
        log.info("   → API       : http://localhost:8000/api/v1/drift/events")
        if not args.daemon:
            break
        log.info(f"💤 Prochain cycle dans {args.interval}s ({args.interval//60} min)...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
