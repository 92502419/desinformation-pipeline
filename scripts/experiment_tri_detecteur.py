#!/usr/bin/env python3
# scripts/experiment_tri_detecteur.py
# Produit le Tableau 4.3 du memoire : comparaison ADWIN / KSWIN / PageHinkley
# (+ score composite pondere) sur 4 scenarios de derive (A abrupt, B graduel,
# C cyclique, D incremental), avec delai de detection moyen, TPR et FPR.
#
# Donnees REELLES : le signal surveille est le p_fake produit par le modele
# Continual-DistilBERT (models/pretrained/) sur des articles reels du jeu de
# test (data/processed/test/test.csv). La "derive" est une variation de la
# proportion d'articles fake dans le flux -> deplacement de la distribution de
# p_fake, exactement comme dans scripts/inject_drift_simulation.py et comme
# surveille en production par spark-app/src/drift_monitor.py.
#
# Aucune donnee generee : seuls des articles reels sont utilises ; seule leur
# mise en ordre (proportion fake au cours du temps) est controlee.
#
# Sorties :
#   reports/tableau_4_3_tri_detecteur.json
#   reports/tableau_4_3_tri_detecteur.md
#   reports/figures/09_tri_detecteur_scenarios.png
#
# Usage :  python scripts/experiment_tri_detecteur.py [--reps 30] [--pool 1600]

import os, sys, json, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only (driver NVML casse)

import numpy as np
import pandas as pd

# --- Parametres des detecteurs individuels (documentation/figure uniquement) :
# IDENTIQUES a spark-app/src/drift_monitor.py
ADWIN_DELTA   = 0.002
KSWIN_WINDOW  = 100
KSWIN_STAT    = KSWIN_WINDOW // 3          # 33
KSWIN_ALPHA   = 0.005
PH_MIN_INST   = 30
PH_DELTA      = 0.005
PH_THRESHOLD  = 50.0
PH_ALPHA      = 1 - 0.0001

HORIZON       = 500       # fenetre (messages) apres le point de derive pour compter une detection
WARMUP        = KSWIN_WINDOW  # on ignore les alarmes avant ce nombre de messages (amorçage)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "spark-app", "src"))
TEST_CSV = os.path.join(ROOT, "data/processed/test/test.csv")
REPORTS  = os.path.join(ROOT, "reports")
FIGS     = os.path.join(REPORTS, "figures")


def build_pfake_pools(pool_per_class: int, seed: int = 0):
    """Calcule p_fake du modele sur un pool d'articles reels welfake+fakenewsnet
    (sources a signal net), separes par vraie classe."""
    import torch
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    df = pd.read_csv(TEST_CSV)
    df = df[df["source"].isin(["welfake", "fakenewsnet"])].copy()
    rng = np.random.default_rng(seed)
    real = df[df.label == 0].sample(min(pool_per_class, (df.label == 0).sum()), random_state=seed)
    fake = df[df.label == 1].sample(min(pool_per_class, (df.label == 1).sum()), random_state=seed + 1)

    tok = DistilBertTokenizerFast.from_pretrained(os.path.join(ROOT, "models/pretrained"))
    mdl = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(ROOT, "models/pretrained")).cpu().eval()

    def infer(frame):
        out = []
        texts = [f"{str(t)[:200]} [SEP] {str(b)[:100]}" for t, b in zip(frame.title, frame.body)]
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], max_length=128, padding="max_length",
                      truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits
            probs = torch.softmax(logits, -1).numpy()[:, 1]
            out.append(probs)
        return np.concatenate(out)

    t0 = time.time()
    p_real = infer(real)
    p_fake = infer(fake)
    print(f"[pool] p_fake calcule sur {len(p_real)} reels + {len(p_fake)} fakes "
          f"en {time.time() - t0:.0f}s | mean(real)={p_real.mean():.3f} mean(fake)={p_fake.mean():.3f}")
    return p_real, p_fake


def make_stream(prop_schedule, p_real, p_fake, rng):
    """Genere un flux de p_fake reels : a l'etape t, on tire un article fake avec
    proba prop_schedule[t], sinon un article reel ; on renvoie son p_fake mesure."""
    xs = np.empty(len(prop_schedule))
    for t, prop in enumerate(prop_schedule):
        if rng.random() < prop:
            xs[t] = p_fake[rng.integers(len(p_fake))]
        else:
            xs[t] = p_real[rng.integers(len(p_real))]
    return xs


def scenario_schedules():
    """Retourne {nom: (prop_schedule (np.array), drift_point (int))} — calque sur
    scripts/inject_drift_simulation.py et la section 3.4.3 du memoire."""
    base = 0.5
    hi = 0.9
    S = {}
    # A - abrupt : palier bas puis saut immediat
    S["A - abrupt"] = (np.concatenate([np.full(1000, base), np.full(1000, hi)]), 1000)
    # B - graduel : palier bas, rampe lineaire 0.5->0.9, palier haut
    ramp = np.linspace(base, hi, 1000)
    S["B - graduel"] = (np.concatenate([np.full(1000, base), ramp, np.full(500, hi)]), 1000)
    # C - cyclique : 4 rafales alternees bas/haut
    S["C - cyclique"] = (np.concatenate([np.full(700, base), np.full(700, hi),
                                         np.full(700, base), np.full(700, hi)]), 700)
    # D - incremental : palier bas puis 10 paliers +0.04
    paliers = np.concatenate([np.full(150, base + 0.04 * (k + 1)) for k in range(10)])
    S["D - incremental"] = (np.concatenate([np.full(1000, base), paliers]), 1000)
    return S


# ----------------------- Detecteurs -----------------------
def run_single_detector(name, xs):
    """Renvoie la liste des indices ou le detecteur signale une derive."""
    from river import drift
    from drift_monitor import binary_entropy, ENTROPY_ADWIN_DELTA
    if name == "ADWIN":
        det = drift.ADWIN(delta=ADWIN_DELTA)
        signal = lambda x: float(x)
    elif name == "KSWIN":
        det = drift.KSWIN(window_size=KSWIN_WINDOW, stat_size=KSWIN_STAT, alpha=KSWIN_ALPHA)
        signal = lambda x: float(x)
    elif name == "PageHinkley":
        det = drift.PageHinkley(min_instances=PH_MIN_INST, delta=PH_DELTA,
                                threshold=PH_THRESHOLD, alpha=PH_ALPHA)
        signal = lambda x: float(x)
    elif name == "EntropyADWIN":
        det = drift.ADWIN(delta=ENTROPY_ADWIN_DELTA)
        signal = lambda x: binary_entropy(float(x))
    else:
        raise ValueError(name)
    alarms = []
    for i, x in enumerate(xs):
        det.update(signal(x))
        if det.drift_detected:
            alarms.append(i)
    return alarms


def run_composite(xs):
    """Utilise directement spark-app/src/drift_monitor.py::DynamicDriftMonitor —
    même code que la production (voir tests/test_shared_modules.py pour la
    philosophie : ne jamais réimplémenter la logique de décision en double)."""
    from drift_monitor import DynamicDriftMonitor
    mon = DynamicDriftMonitor()
    alarms, was_active = [], False
    for i, x in enumerate(xs):
        mon.update(confidence=float(x))
        if mon.drift_active and not was_active:
            alarms.append(i)
        was_active = mon.drift_active
    return alarms


DETECTORS = ["ADWIN", "KSWIN", "PageHinkley", "EntropyADWIN", "Composite"]


def evaluate(reps, pool_per_class, seed0=0):
    p_real, p_fake = build_pfake_pools(pool_per_class, seed=seed0)
    scen = scenario_schedules()
    results = {}
    for sname, (sched, dpoint) in scen.items():
        per_det = {d: {"delays": [], "detected": 0, "false_alarms": 0} for d in DETECTORS}
        for r in range(reps):
            rng = np.random.default_rng(1000 * seed0 + r)
            xs = make_stream(sched, p_real, p_fake, rng)
            for d in DETECTORS:
                alarms = run_composite(xs) if d == "Composite" else run_single_detector(d, xs)
                # faux positifs : alarme dans la zone stationnaire [WARMUP, dpoint)
                fa = any(WARMUP <= i < dpoint for i in alarms)
                per_det[d]["false_alarms"] += int(fa)
                # detection : 1re alarme dans [dpoint, dpoint+HORIZON)
                post = [i for i in alarms if dpoint <= i < dpoint + HORIZON]
                if post:
                    per_det[d]["detected"] += 1
                    per_det[d]["delays"].append(post[0] - dpoint)
        summary = {}
        for d in DETECTORS:
            dd = per_det[d]
            delays = np.array(dd["delays"], dtype=float)
            summary[d] = {
                "tpr": round(dd["detected"] / reps, 3),
                "fpr": round(dd["false_alarms"] / reps, 3),
                "delay_mean": round(float(delays.mean()), 1) if len(delays) else None,
                "delay_median": round(float(np.median(delays)), 1) if len(delays) else None,
                "n_detected": dd["detected"],
                "reps": reps,
            }
        results[sname] = {"drift_point": dpoint, "detectors": summary}
    return results, (p_real, p_fake, scen)


def make_figure(p_real, p_fake, scen, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (sname, (sched, dpoint)) in zip(axes.ravel(), scen.items()):
        xs = make_stream(sched, p_real, p_fake, rng)
        # moyenne glissante pour lisibilite
        w = 50
        mov = np.convolve(xs, np.ones(w) / w, mode="valid")
        ax.plot(range(len(xs)), xs, ".", ms=1.5, alpha=0.25, color="#95A5A6")
        ax.plot(range(w - 1, len(xs)), mov, color="#1565C0", lw=1.8, label="p_fake (moy. 50)")
        ax.plot(range(len(sched)), sched, color="#E67E22", lw=1.2, ls="--",
                label="proportion fake cible")
        ax.axvline(dpoint, color="#C0392B", lw=1.5, label="point de derive")
        colors = {"ADWIN": "#E74C3C", "KSWIN": "#3498DB", "PageHinkley": "#27AE60",
                  "EntropyADWIN": "#8E44AD"}
        for d, c in colors.items():
            al = run_single_detector(d, xs)
            al = [i for i in al if i >= WARMUP]
            ax.plot(al, [1.02] * len(al), "|", color=c, ms=12, mew=2, label=f"{d} alarme")
        ax.set_title(sname); ax.set_ylim(-0.03, 1.08); ax.set_xlabel("message"); ax.set_ylabel("p_fake")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("Tri-detecteur de Concept Drift — un tirage par scenario (donnees reelles)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    print(f"[fig] {out_png}")


def write_markdown(results, meta, out_md):
    dp = meta['detector_params']
    lines = []
    lines.append("# Tableau 4.3 — Performances du tri-detecteur de Concept Drift\n")
    lines.append(f"_Genere par `scripts/experiment_tri_detecteur.py` le "
                 f"{time.strftime('%Y-%m-%d')} — {meta['reps']} repetitions par scenario, "
                 f"pool de {meta['pool_per_class']} articles reels/classe (welfake + fakenewsnet)._\n")
    lines.append("Signal surveille : `p_fake` du modele Continual-DistilBERT sur des articles "
                 "**reels** du jeu de test. La derive = variation de la proportion d'articles fake "
                 "dans le flux (calque sur `scripts/inject_drift_simulation.py` et la section 3.4.3). "
                 f"Detection comptee si une alarme survient dans les {HORIZON} messages suivant le "
                 "point de derive ; faux positif = alarme dans la zone stationnaire. "
                 f"Parametres identiques a `spark-app/src/drift_monitor.py` "
                 f"(ADWIN δ={ADWIN_DELTA}, KSWIN α={KSWIN_ALPHA} fenetre {KSWIN_WINDOW}, "
                 f"PageHinkley δ={PH_DELTA} λ={PH_THRESHOLD}, "
                 f"EntropyADWIN δ={dp['ENTROPY_ADWIN_DELTA']}, "
                 f"composite seuil {dp['COMPOSITE_THR']}, poids {dp['WEIGHTS']}, "
                 f"remanence {dp['SIGNAL_HOLD']}). River {meta['river_version']}.\n")
    lines.append("| Scenario | Detecteur | Delai moyen (msg) | Delai median | TPR | FPR | n detect. |")
    lines.append("|---|---|---|---|---|---|---|")
    for sname, blob in results.items():
        for d in DETECTORS:
            s = blob["detectors"][d]
            dm = "-" if s["delay_mean"] is None else f"{s['delay_mean']:.0f}"
            dmd = "-" if s["delay_median"] is None else f"{s['delay_median']:.0f}"
            lines.append(f"| {sname} | {d} | {dm} | {dmd} | {s['tpr']:.2f} | {s['fpr']:.2f} | "
                         f"{s['n_detected']}/{s['reps']} |")
    lines.append("")
    lines.append("![Tri-detecteur par scenario](figures/09_tri_detecteur_scenarios.png)\n")
    lines.append("## Lecture\n")
    lines.append("- **Delai moyen** : nombre de messages entre le point de derive reel et la "
                 "premiere alarme du detecteur (sur les repetitions ou il a detecte).")
    lines.append("- **TPR** : fraction des repetitions ou le detecteur a signale la derive dans "
                 f"la fenetre de {HORIZON} messages.")
    lines.append("- **FPR** : fraction des repetitions avec au moins une fausse alarme avant la "
                 "derive (zone stationnaire).")
    w = dp['WEIGHTS']
    lines.append(f"- Le **score composite** ({w['ADWIN']}·ADWIN + {w['KSWIN']}·KSWIN + "
                 f"{w['PageHinkley']}·PageHinkley, remanence de {dp['SIGNAL_HOLD']} messages, "
                 f"seuil {dp['COMPOSITE_THR']}), complete par **EntropyADWIN en OU logique** "
                 "(canal independant, n'entre pas dans la ponderation — une premiere version "
                 "ponderee degradait le TPR des scenarios B/D, cf. drift_monitor.py), est la "
                 "strategie reellement utilisee en production (spark-app/src/drift_monitor.py).")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[md] {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--pool", type=int, default=1600, help="articles reels par classe pour le pool p_fake")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    import river
    from drift_monitor import DynamicDriftMonitor, DRIFT_THRESH, ENTROPY_ADWIN_DELTA
    _ref_mon = DynamicDriftMonitor()  # juste pour lire les constantes reellement utilisees
    WEIGHTS, SIGNAL_HOLD, COMPOSITE_THR = _ref_mon.weights, _ref_mon.SIGNAL_HOLD, DRIFT_THRESH
    t0 = time.time()
    results, (p_real, p_fake, scen) = evaluate(args.reps, args.pool, args.seed)
    meta = {"reps": args.reps, "pool_per_class": int(min(args.pool, len(p_real))),
            "river_version": river.__version__, "seconds": round(time.time() - t0, 1),
            "horizon": HORIZON, "warmup": WARMUP,
            "detector_params": {"ADWIN_DELTA": ADWIN_DELTA, "KSWIN_WINDOW": KSWIN_WINDOW,
                                "KSWIN_ALPHA": KSWIN_ALPHA, "PH_DELTA": PH_DELTA,
                                "PH_THRESHOLD": PH_THRESHOLD, "ENTROPY_ADWIN_DELTA": ENTROPY_ADWIN_DELTA,
                                "WEIGHTS": WEIGHTS,
                                "SIGNAL_HOLD": SIGNAL_HOLD, "COMPOSITE_THR": COMPOSITE_THR}}
    with open(os.path.join(REPORTS, "tableau_4_3_tri_detecteur.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    make_figure(p_real, p_fake, scen, os.path.join(FIGS, "09_tri_detecteur_scenarios.png"))
    write_markdown(results, meta, os.path.join(REPORTS, "tableau_4_3_tri_detecteur.md"))
    print(f"\nTermine en {meta['seconds']}s.")
    for sname, blob in results.items():
        print(f"\n{sname}  (point de derive={blob['drift_point']})")
        for d in DETECTORS:
            s = blob["detectors"][d]
            print(f"  {d:12s} delai={s['delay_mean']}  TPR={s['tpr']}  FPR={s['fpr']}")


if __name__ == "__main__":
    main()
