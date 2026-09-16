#!/usr/bin/env python3
# scripts/experiment_derive_opportunite.py
# Experience centrale du memoire : "la derive, correctement geree, est une
# opportunite d'apprentissage" (hypothese H3, sections 3.4.3 / 3.5.1 / 4.1.3 / 4.4).
#
# Scenario (calque sur inject_drift_simulation.py, scenario A/B) : une CAMPAGNE
# de desinformation monte en puissance -> derive de PROPORTION (le tri-detecteur
# `drift_monitor.py` la voit et fait passer le LR de 1e-5 a 5e-5 AUTOMATIQUEMENT)
# combinee a une derive de CONCEPT (la vague de fake est tiree du sous-corpus
# LIAR : declarations politiques courtes, terrain faible du modele -> le F1 chute
# vraiment, il y a donc quelque chose a "recuperer").
#
# On compare, sur ce MEME flux, en protocole prequential (test-then-train) :
#   - modele STATIQUE (fige)
#   - modele DYNAMIQUE (Continual-DistilBERT : online learning calque 1:1 sur
#     spark-app/src/nlp_classifier.py::online_update, LR pilote par le tri-detecteur)
# + une SONDE D'OUBLI : evaluation periodique des deux modeles sur un jeu
#   welfake/fnn tenu a l'ecart du flux (verifie que le reservoir replay limite
#   l'oubli catastrophique).
#
# Donnees REELLES uniquement (data/processed/test/test.csv). Seul l'ORDRE et la
# PROPORTION fake/reel au cours du temps sont controles. Aucune donnee generee.
#
# Sorties : reports/experience_derive_opportunite.{json,md}
#           reports/figures/10_derive_opportunite.png
#
# Usage : python scripts/experiment_derive_opportunite.py
#           [--pre 800 --drift 1400 --pfake_campaign 0.80 --chunk 40 --seed 0]

import os, sys, json, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "spark-app", "src"))
TEST_CSV = os.path.join(ROOT, "data/processed/test/test.csv")
PRETRAINED = os.path.join(ROOT, "models/pretrained")
REPORTS = os.path.join(ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")

# --- online : IDENTIQUE a spark-app/src/nlp_classifier.py / online_trainer.py ---
LR_BASE = 1e-5
LR_DRIFT = 5e-5
RESERVOIR_PER_CLASS = 2500
MAX_EACH = 4
REPLAY_PER_CLASS = 2
ROLL_WIN = 200
PROBE_EVERY = 300      # messages
PROBE_SIZE = 200       # articles welfake/fnn tenus a l'ecart


def load_model():
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    tok = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    mdl = DistilBertForSequenceClassification.from_pretrained(PRETRAINED).cpu().eval()
    return tok, mdl


def infer_pfake(tok, mdl, texts, bs=48):
    import torch
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], max_length=128, padding="max_length",
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.softmax(mdl(**enc).logits, -1).numpy()[:, 1])
    return np.concatenate(out)


def build_stream(pre, drift, pfake_campaign, seed):
    """P0 stationnaire (welfake/fnn, 50% fake) -> campagne abrupte (pfake_campaign
    de fake tires de LIAR, le reste = reel welfake/fnn)."""
    df = pd.read_csv(TEST_CSV)
    strong = df[df.source.isin(["welfake", "fakenewsnet"])]
    liar = df[df.source == "liar"]
    rng = np.random.default_rng(seed)

    # jeu de sonde d'oubli : welfake/fnn tenus a l'ecart du flux
    probe_idx = rng.choice(strong.index.values, size=PROBE_SIZE, replace=False)
    probe = df.loc[probe_idx]
    stream_strong = strong.drop(index=probe_idx)

    s_fake = stream_strong[stream_strong.label == 1].index.values
    s_real = stream_strong[stream_strong.label == 0].index.values
    l_fake = liar[liar.label == 1].index.values
    l_real = liar[liar.label == 0].index.values

    rows_idx = []
    # P0 : 50% fake, tout welfake/fnn
    for _ in range(pre):
        rows_idx.append(rng.choice(s_fake if rng.random() < 0.5 else s_real))
    # campagne : pfake_campaign de fake (LIAR), le reste reel (welfake/fnn)
    for _ in range(drift):
        if rng.random() < pfake_campaign:
            rows_idx.append(rng.choice(l_fake))
        else:
            rows_idx.append(rng.choice(s_real))
    s = df.loc[rows_idx].reset_index(drop=True)
    s["text"] = [f"{str(t)[:200]} [SEP] {str(b)[:100]}" for t, b in zip(s.title, s.body)]
    probe = probe.reset_index(drop=True)
    probe["text"] = [f"{str(t)[:200]} [SEP] {str(b)[:100]}" for t, b in zip(probe.title, probe.body)]
    return s, probe


class OnlineModel:
    """Calque de spark-app/src/nlp_classifier.py::ContinualDistilBERT (partie PyTorch)."""

    def __init__(self, tok, mdl):
        import torch
        from torch.optim import SGD
        self.torch = torch
        self.tok, self.m = tok, mdl
        self.opt = SGD(self.m.parameters(), lr=LR_BASE, momentum=0.9, weight_decay=0.01)
        self.cur_lr = LR_BASE
        self.res_fake, self.res_real = [], []
        self.n_fake = self.n_real = 0

    def reservoir_update(self, text, label):
        if label == 1:
            self.n_fake += 1; buf, n = self.res_fake, self.n_fake
        else:
            self.n_real += 1; buf, n = self.res_real, self.n_real
        if len(buf) < RESERVOIR_PER_CLASS:
            buf.append((text, label))
        else:
            j = np.random.randint(0, n)
            if j < RESERVOIR_PER_CLASS:
                buf[j] = (text, label)

    def online_update(self, texts, labels, lr):
        torch = self.torch
        if lr != self.cur_lr:
            for pg in self.opt.param_groups:
                pg["lr"] = lr
            self.cur_lr = lr
        fake_in = [(t, l) for t, l in zip(texts, labels) if l == 1]
        real_in = [(t, l) for t, l in zip(texts, labels) if l == 0]
        n_each = min(len(fake_in), len(real_in), MAX_EACH)
        selected = (fake_in[:n_each] + real_in[:n_each]) if n_each > 0 \
            else list(zip(texts, labels))[:MAX_EACH]
        tr_t = [s[0] for s in selected]; tr_l = [s[1] for s in selected]
        rp = min(REPLAY_PER_CLASS, len(self.res_fake), len(self.res_real))
        if rp > 0:
            fi = np.random.choice(len(self.res_fake), rp, replace=False)
            ri = np.random.choice(len(self.res_real), rp, replace=False)
            rep = [self.res_fake[i] for i in fi] + [self.res_real[i] for i in ri]
            tr_t += [r[0] for r in rep]; tr_l += [r[1] for r in rep]
        n_f = sum(1 for l in tr_l if l == 1); n_r = len(tr_l) - n_f
        cw = torch.tensor([n_f / len(tr_l), n_r / len(tr_l)], dtype=torch.float) if (n_f and n_r) else None
        self.m.train()
        enc = self.tok(tr_t, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        yt = torch.tensor(tr_l, dtype=torch.long)
        self.opt.zero_grad()
        loss = torch.nn.CrossEntropyLoss(weight=cw)(self.m(**enc).logits, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
        self.opt.step()
        lv = float(loss.item())
        self.opt.zero_grad(set_to_none=True)
        self.m.eval()
        for t, l in zip(texts, labels):
            self.reservoir_update(t, l)
        return lv


def f1_macro(yt, yp):
    from sklearn.metrics import f1_score
    return f1_score(yt, yp, average="macro", zero_division=0)


def rolling_f1(yt, yp, win=ROLL_WIN):
    yt, yp = np.array(yt), np.array(yp)
    o = np.full(len(yt), np.nan)
    for i in range(len(yt)):
        o[i] = f1_macro(yt[max(0, i - win + 1):i + 1], yp[max(0, i - win + 1):i + 1])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=800)
    ap.add_argument("--drift", type=int, default=1400)
    ap.add_argument("--pfake_campaign", type=float, default=0.80)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force_lr", type=float, default=0.0,
                    help="si >0 : LR force a cette valeur des le debut de la campagne "
                         "(simule un tri-detecteur qui aurait correctement declenche). "
                         "Sert a caracteriser le MECANISME d'adaptation independamment de "
                         "la sensibilite du detecteur.")
    ap.add_argument("--tag", default="", help="suffixe de nom de fichier de sortie")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    TAG = ("_" + args.tag) if args.tag else ""
    JSON_OUT = os.path.join(REPORTS, f"experience_derive_opportunite{TAG}.json")
    MD_OUT   = os.path.join(REPORTS, f"experience_derive_opportunite{TAG}.md")
    PNG_OUT  = os.path.join(FIGS, f"10_derive_opportunite{TAG}.png")
    PNG_REL  = f"figures/10_derive_opportunite{TAG}.png"
    import river
    from drift_monitor import DynamicDriftMonitor

    t0 = time.time()
    stream, probe = build_stream(args.pre, args.drift, args.pfake_campaign, args.seed)
    N = len(stream)
    texts = stream["text"].tolist()
    y = stream["label"].to_numpy()
    py = probe["label"].to_numpy()
    ptexts = probe["text"].tolist()
    dpoint = args.pre
    early_end = args.pre + min(400, args.drift // 3)
    print(f"[stream] N={N} | P0=[0,{args.pre}) campagne=[{args.pre},{N}) "
          f"p_fake_campagne={args.pfake_campaign} | sonde welfake/fnn n={len(probe)} | seed={args.seed}", flush=True)

    tok, base = load_model()
    print("[static] inference flux + sonde ...", flush=True)
    p_static = infer_pfake(tok, base, texts)
    pred_static = (p_static >= 0.5).astype(int)
    probe_static_f1 = round(f1_macro(py, (infer_pfake(tok, base, ptexts) >= 0.5).astype(int)), 4)
    print(f"[static] F1 global flux={f1_macro(y, pred_static):.4f} | F1 sonde={probe_static_f1} "
          f"({time.time()-t0:.0f}s)", flush=True)

    tok2, mdl2 = load_model()
    dyn = OnlineModel(tok2, mdl2)
    mon = DynamicDriftMonitor()
    pred_dyn = np.empty(N, dtype=int)
    lr_hist = np.empty(N, dtype=float)
    drift_active = np.zeros(N, dtype=int)
    probe_curve = []   # (message_index, f1_dynamic_probe)
    losses = []
    first_detection = None
    print("[dynamic] prequential ...", flush=True)
    for start in range(0, N, args.chunk):
        end = min(N, start + args.chunk)
        ch_t, ch_y = texts[start:end], y[start:end].tolist()
        pf = infer_pfake(tok2, mdl2, ch_t)
        pred_dyn[start:end] = (pf >= 0.5).astype(int)
        for v in pf:
            mon.update(confidence=float(v))
        lr = mon.get_recommended_lr()
        if args.force_lr > 0 and start >= dpoint:
            lr = args.force_lr        # signal oracle : detecteur suppose declenche
        lr_hist[start:end] = lr
        drift_active[start:end] = int(mon.is_drift_active())
        if first_detection is None and mon.is_drift_active():
            first_detection = end
        losses.append(dyn.online_update(ch_t, ch_y, lr=lr))
        # sonde d'oubli
        if (start // args.chunk) % max(1, PROBE_EVERY // args.chunk) == 0:
            pf1 = round(f1_macro(py, (infer_pfake(tok2, mdl2, ptexts) >= 0.5).astype(int)), 4)
            probe_curve.append([end, pf1])
        if (start // args.chunk) % 6 == 0:
            d = end
            print(f"  {d}/{N} lr={lr:.0e} drift={mon.is_drift_active()} loss={losses[-1]:.3f} "
                  f"F1dyn(roll)={f1_macro(y[max(0,d-ROLL_WIN):d], pred_dyn[max(0,d-ROLL_WIN):d]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # sonde finale
    pf1_end = round(f1_macro(py, (infer_pfake(tok2, mdl2, ptexts) >= 0.5).astype(int)), 4)
    probe_curve.append([N, pf1_end])

    seg = {"P0 (pre-campagne, welfake/fnn)": (0, args.pre),
           "P1 (debut campagne)": (args.pre, early_end),
           "P2 (campagne etablie, apres adaptation)": (early_end, N)}
    per_phase = {}
    for name, (a, b) in seg.items():
        per_phase[name] = {
            "n": int(b - a),
            "f1_static": round(f1_macro(y[a:b], pred_static[a:b]), 4),
            "f1_dynamic": round(f1_macro(y[a:b], pred_dyn[a:b]), 4),
        }
    f1d_p0 = per_phase["P0 (pre-campagne, welfake/fnn)"]["f1_dynamic"]
    f1d_p2 = per_phase["P2 (campagne etablie, apres adaptation)"]["f1_dynamic"]
    f1s_p2 = per_phase["P2 (campagne etablie, apres adaptation)"]["f1_static"]
    verdict = {
        "detecteur_a_declenche": first_detection is not None,
        "index_1re_detection": first_detection,
        "delai_detection_msg": (first_detection - dpoint) if first_detection else None,
        "gain_dynamique_vs_statique_P2_pts_f1": round((f1d_p2 - f1s_p2) * 100, 2),
        "recuperation_dynamique_P2_moins_P0_pts_f1": round((f1d_p2 - f1d_p0) * 100, 2),
        "oubli_sonde_pts_f1": round((probe_static_f1 - pf1_end) * 100, 2),
        "f1_sonde_statique": probe_static_f1,
        "f1_sonde_dynamique_fin": pf1_end,
        "derive_est_opportunite": bool(f1d_p2 > f1s_p2 and f1d_p2 >= f1d_p0 - 0.02),
    }
    meta = {"seed": args.seed, "N": N, "pre": args.pre, "drift": args.drift,
            "pfake_campaign": args.pfake_campaign, "chunk": args.chunk,
            "phases": {k: list(v) for k, v in seg.items()}, "drift_point": dpoint,
            "lr_base": LR_BASE, "lr_drift": LR_DRIFT, "roll_win": ROLL_WIN,
            "n_online_updates": len(losses), "mean_loss": round(float(np.mean(losses)), 4),
            "river_version": river.__version__, "seconds": round(time.time() - t0, 1), "force_lr": args.force_lr,
            "note": "Bras dynamique = calque de spark-app/src/nlp_classifier.py::online_update. "
                    "LR pilote automatiquement par le tri-detecteur drift_monitor.py sur le p_fake du flux."}

    with open(JSON_OUT, "w") as f:
        json.dump({"meta": meta, "per_phase": per_phase, "probe_curve": probe_curve,
                   "verdict": verdict}, f, indent=2, ensure_ascii=False)

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs, rd = rolling_f1(y, pred_static), rolling_f1(y, pred_dyn)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(rs, color="#7F8C8D", lw=2, label="F1 STATIQUE (glissant 200)")
    ax1.plot(rd, color="#1565C0", lw=2, label="F1 DYNAMIQUE (glissant 200)")
    pc = np.array(probe_curve)
    ax1.plot(pc[:, 0], pc[:, 1], "o-", color="#8E44AD", ms=4, lw=1.4,
             label="sonde d'oubli — F1 dynamique sur welfake/fnn tenu a l'ecart")
    ax1.axhline(probe_static_f1, color="#8E44AD", ls=":", lw=1, alpha=0.7,
                label=f"sonde — F1 statique ({probe_static_f1:.3f})")
    ax1.axvspan(0, args.pre, color="#EAF7EA", alpha=0.6)
    ax1.axvspan(args.pre, N, color="#FBECEC", alpha=0.55)
    ax1.axvline(dpoint, color="#C0392B", ls="--", lw=1.5, label="debut de la campagne")
    if first_detection:
        ax1.axvline(first_detection, color="#E67E22", ls="-.", lw=1.5,
                    label=f"1re detection (delai {first_detection - dpoint} msg)")
    ax1.set_ylim(0.3, 1.02); ax1.set_ylabel("F1-macro glissant")
    ax1.set_title("Derive = opportunite : dynamique vs statique — campagne de fake LIAR "
                  "(derive proportion + concept), donnees reelles")
    ax1.legend(loc="lower left", fontsize=8)
    ax2.plot(lr_hist, color="#E67E22", lw=1.5)
    ax2.set_yscale("log"); ax2.set_ylabel("LR online"); ax2.set_xlabel("message (flux)")
    ax2.axvline(dpoint, color="#C0392B", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=115)
    print(f"[fig] {PNG_OUT}", flush=True)

    L = ["# Experience — la derive comme opportunite d'apprentissage\n",
         f"_Genere par `scripts/experiment_derive_opportunite.py` le {time.strftime('%Y-%m-%d')} "
         f"(seed {args.seed}, {N} messages, {meta['n_online_updates']} MAJ online, {meta['seconds']:.0f}s CPU)._\n",
         "Scenario : une **campagne de desinformation** monte abruptement en puissance a "
         f"t={dpoint} (la proportion d'articles fake passe de 50 % a {int(args.pfake_campaign*100)} %). "
         "Les fake de la campagne sont tires du sous-corpus **LIAR** (declarations politiques courtes, "
         "terrain faible du modele) : la derive est donc a la fois de **proportion** (vue par le "
         "tri-detecteur `drift_monitor.py`, qui fait passer le LR de 1e-5 a 5e-5 automatiquement) et "
         "de **concept** (le F1 chute). Protocole prequential (test-then-train, section 3.5.1). "
         "Bras dynamique = `spark-app/src/nlp_classifier.py` a l'identique. Articles **reels** "
         "(data/processed/test/test.csv) ; seul l'ordre/la proportion sont controles.\n",
         "| Phase | n | F1-macro STATIQUE | F1-macro DYNAMIQUE |",
         "|---|---|---|---|"]
    for name, d in per_phase.items():
        L.append(f"| {name} | {d['n']} | {d['f1_static']:.4f} | {d['f1_dynamic']:.4f} |")
    det_txt = "a declenche" if verdict["detecteur_a_declenche"] else "n'a PAS declenche"
    det_delay = (f" (delai {verdict['delai_detection_msg']} messages apres le debut de la campagne)"
                 if verdict["delai_detection_msg"] is not None else "")
    L += ["",
          f"- **Tri-detecteur** : {det_txt}{det_delay}.",
          f"- **Gain dynamique vs statique en P2 (campagne etablie)** : "
          f"**{verdict['gain_dynamique_vs_statique_P2_pts_f1']:+.2f} points de F1-macro**.",
          f"- **Recuperation du modele dynamique en P2 par rapport a P0** : "
          f"{verdict['recuperation_dynamique_P2_moins_P0_pts_f1']:+.2f} points.",
          f"- **Oubli catastrophique** (sonde welfake/fnn tenue a l'ecart) : F1 statique "
          f"{verdict['f1_sonde_statique']:.4f} -> F1 dynamique fin {verdict['f1_sonde_dynamique_fin']:.4f} "
          f"(**{verdict['oubli_sonde_pts_f1']:+.2f} pts** ; positif = oubli).",
          f"- Paradigme \"derive = opportunite\" verifie ? "
          f"**{'OUI' if verdict['derive_est_opportunite'] else 'NON'}** "
          f"(F1 dyn. P2 > F1 statique P2 et F1 dyn. P2 >= F1 dyn. P0 - 2 pts).",
          "", f"![Derive = opportunite]({PNG_REL})\n"]
    with open(MD_OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[md] {MD_OUT}", flush=True)

    print("\n==== RESUME ====", flush=True)
    for name, d in per_phase.items():
        print(f"  {name:42s} F1 stat={d['f1_static']:.4f}  F1 dyn={d['f1_dynamic']:.4f}")
    print(f"  detection={verdict['detecteur_a_declenche']} delai={verdict['delai_detection_msg']} | "
          f"gain P2={verdict['gain_dynamique_vs_statique_P2_pts_f1']:+.2f} | "
          f"oubli={verdict['oubli_sonde_pts_f1']:+.2f} | "
          f"opportunite={verdict['derive_est_opportunite']}")


if __name__ == "__main__":
    main()
