#!/usr/bin/env python3
# scripts/inject_drift_simulation.py — Simulation de Concept Drift pour la démonstration
# de soutenance (voir Documentation_Technique_..._v9.docx, ÉTAPE 15, Acte 4).
#
# v2.4 : 4 scénarios (A=abrupt, B=graduel, C=cyclique, D=incrémental) + récupération
# automatique après une fenêtre d'observation (`visualization_window`), pour ne plus
# jamais laisser la simulation tourner en démon 24h/24 comme dans une version antérieure
# (incident documenté : 83 % de la base MongoDB polluée par des articles de simulation).
#
# Utilisable en CLI (démo manuelle) ou importé par api/src/routers/drift.py
# (endpoints POST /api/v1/drift/inject et /api/v1/drift/recover).

import argparse
import json
import random
import time

from confluent_kafka import Producer

FAKE_TITLES_FR = [
    'URGENT — Découverte choquante sur les ondes 5G dissimulée par les autorités !',
    'EXCLUSIF : Le remède miracle censuré que Big Pharma veut effacer',
    'ALERTE : Preuve irréfutable du complot mondial enfin révélée',
    'SCANDALE : Ce que le gouvernement vous cache depuis des années',
    'BREAKING : Fuite de documents secrets explosifs sur la vaccination',
    'INCROYABLE : Scientifiques réduits au silence après leur découverte',
    'CHOC : La vérité sur ce qui se passe vraiment en ce moment',
    'URGENT : Ceci va changer votre vie pour toujours — partagez !',
]

REAL_TITLES_FR = [
    "Le conseil des ministres a adopté un nouveau plan d'investissement agricole",
    'La Banque centrale maintient son taux directeur inchangé ce trimestre',
    'Un sommet régional se tiendra la semaine prochaine sur la sécurité alimentaire',
    "L'OMS publie son rapport annuel sur la couverture vaccinale en Afrique",
    'Le ministère des Finances présente le projet de budget 2027',
    'Une délégation diplomatique conclut un accord de coopération économique',
]


def _make_msg(title: str, source: str, category: str, tone_range=(-1.0, 1.0)) -> dict:
    return {
        'id': f'drift_sim_{int(time.time()*1000)}_{random.randint(0, 999999)}',
        'title': title,
        'body': (
            'Contenu simulant une nouvelle campagne de désinformation.'
            if category == 'suspicious' else
            "Communiqué de presse officiel diffusé à des fins de démonstration."
        ),
        'url': f'https://simulation.test/article/{random.randint(0, 999999)}',
        'source': source,
        'source_category': category,
        'language': 'fr',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'gdelt_tone': round(random.uniform(*tone_range), 2),
    }


def _send(producer: Producer, topic: str, msg: dict):
    producer.produce(topic, key=msg['id'], value=json.dumps(msg, ensure_ascii=False).encode('utf-8'))


def run_recovery(producer: Producer, topic: str = 'raw-news-stream', n_articles: int = 100):
    """Envoie des articles réels pour ramener le modèle et les statistiques à la normale."""
    print(f'[recovery] Envoi de {n_articles} articles réels pour rééquilibrage...')
    for i in range(n_articles):
        msg = _make_msg(random.choice(REAL_TITLES_FR), 'SimulationRecovery', 'reliable', tone_range=(0.0, 3.0))
        _send(producer, topic, msg)
        if i % 20 == 0:
            producer.flush()
    producer.flush()
    print('[recovery] Terminé — le score de drift devrait redescendre progressivement.')


def run_scenario(scenario: str, broker: str = 'localhost:9092', topic: str = 'raw-news-stream',
                  n: int = 200, rate: float = 0.1, with_recovery: bool = True,
                  visualization_window: int = 300):
    """Injecte un scénario de drift (A/B/C/D), attend `visualization_window` secondes
    d'observation, puis lance automatiquement la récupération si `with_recovery`."""
    producer = Producer({'bootstrap.servers': broker})
    scenario = scenario.upper()
    print(f"Scénario {scenario} — injection vers {broker}/{topic}")
    print('Surveiller Grafana (localhost:3000) → Panneau Score Drift')

    if scenario == 'A':  # abrupt : rafale immédiate, 100% fake
        print(f'[A - abrupt] {n} articles fake envoyés sans délai')
        for i in range(n):
            _send(producer, topic, _make_msg(random.choice(FAKE_TITLES_FR), 'SimulationDrift', 'suspicious', (-8.0, -3.0)))
            if i % 10 == 0:
                producer.flush()
                print(f'  [{i+1}/{n}]')
        producer.flush()

    elif scenario == 'B':  # graduel : proportion de fake croît linéairement de 0 à 100%
        print(f'[B - graduel] {n} articles, proportion de fake croissante')
        for i in range(n):
            p_fake = i / max(1, n - 1)
            is_fake = random.random() < p_fake
            msg = (_make_msg(random.choice(FAKE_TITLES_FR), 'SimulationDrift', 'suspicious', (-8.0, -3.0))
                   if is_fake else
                   _make_msg(random.choice(REAL_TITLES_FR), 'SimulationDrift', 'reliable', (0.0, 3.0)))
            _send(producer, topic, msg)
            if i % 10 == 0:
                producer.flush()
                print(f'  [{i+1}/{n}] p_fake_cible={p_fake:.2f}')
            time.sleep(rate)
        producer.flush()

    elif scenario == 'C':  # cyclique : alternance de rafales fake / réel
        print(f'[C - cyclique] {n} articles en 4 rafales alternées')
        burst = max(1, n // 4)
        for cycle in range(4):
            is_fake_burst = (cycle % 2 == 0)
            for i in range(burst):
                msg = (_make_msg(random.choice(FAKE_TITLES_FR), 'SimulationDrift', 'suspicious', (-8.0, -3.0))
                       if is_fake_burst else
                       _make_msg(random.choice(REAL_TITLES_FR), 'SimulationDrift', 'reliable', (0.0, 3.0)))
                _send(producer, topic, msg)
                time.sleep(rate)
            producer.flush()
            print(f'  Rafale {cycle+1}/4 ({"fake" if is_fake_burst else "réel"}) envoyée')

    elif scenario == 'D':  # incrémental : petits paliers successifs, +10 % de fake à chaque palier
        print(f'[D - incrémental] {n} articles en paliers de 10%')
        n_steps = 10
        per_step = max(1, n // n_steps)
        for step in range(n_steps):
            p_fake = (step + 1) / n_steps
            for i in range(per_step):
                is_fake = random.random() < p_fake
                msg = (_make_msg(random.choice(FAKE_TITLES_FR), 'SimulationDrift', 'suspicious', (-8.0, -3.0))
                       if is_fake else
                       _make_msg(random.choice(REAL_TITLES_FR), 'SimulationDrift', 'reliable', (0.0, 3.0)))
                _send(producer, topic, msg)
                time.sleep(rate)
            producer.flush()
            print(f'  Palier {step+1}/{n_steps} — p_fake={p_fake:.0%}')

    else:
        raise ValueError(f'Scénario inconnu : {scenario} (attendu : A, B, C ou D)')

    print(f'Injection terminée. Fenêtre de visualisation : {visualization_window}s')
    print('Vérifier : curl http://localhost:8000/api/v1/drift/events')

    if with_recovery:
        print(f'Attente de {visualization_window}s avant récupération automatique...')
        time.sleep(visualization_window)
        run_recovery(producer, topic, n_articles=100)
    else:
        print('Récupération automatique désactivée (with_recovery=False) — '
              'lancer manuellement : POST /api/v1/drift/recover')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation de Concept Drift — démo soutenance')
    parser.add_argument('--broker', default='localhost:9092')
    parser.add_argument('--topic', default='raw-news-stream')
    parser.add_argument('--scenario', default='B', choices=['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd'])
    parser.add_argument('--n', type=int, default=200, help='Nb articles à injecter')
    parser.add_argument('--rate', type=float, default=0.1, help='Délai entre messages (sec)')
    parser.add_argument('--with_recovery', action='store_true', default=True)
    parser.add_argument('--no-recovery', dest='with_recovery', action='store_false')
    parser.add_argument('--visualization_window', type=int, default=300)
    args = parser.parse_args()

    run_scenario(args.scenario, broker=args.broker, topic=args.topic, n=args.n, rate=args.rate,
                 with_recovery=args.with_recovery, visualization_window=args.visualization_window)
