"""Phase 1 label-extraction core: prompt, vocabularies, parsing, calibration.

Pure-Python, no GPU deps. Tested locally; the Kaggle notebook embeds this
file verbatim (single self-contained notebook per release convention).
"""

import json
import re

# Short JSON keys -> official train.csv label names
KEY2LABEL = {
    "acl": "ACL",
    "mcl": "MCL",
    "mm": "Medial Meniscus",
    "lm": "Lateral Meniscus",
    "moa": "Medial OA",
    "loa": "Lateral OA",
    "pfoa": "PF OA",
    "effusion": "Effusion",
    "synovitis": "Synovitis",
    "bakers": "Baker's",
    "contusion": "Contusion",
    "fracture": "Fracture",
}
LABEL2KEY = {v: k for k, v in KEY2LABEL.items()}
KEYS = list(KEY2LABEL.keys())

# term -> rank (higher = more abnormal). "not_mentioned" is handled separately
# via the per-label prior; it is NOT part of the ordinal scale.
VOCAB = {
    "acl": {
        "finding": "anterior cruciate ligament (ACL)",
        "terms": {
            "intact": 0,
            "mucoid_degeneration": 1,
            "reconstructed": 1,
            "sprain": 2,
            "partial_tear": 3,
            "complete_tear": 4,
        },
        "notes": ("sprain = low-grade injury, fibers continuous; "
                  "complete_tear = full-thickness disruption/rupture; "
                  "reconstructed = ACL graft present (post-surgical)"),
    },
    "mcl": {
        "finding": "medial collateral ligament (MCL)",
        "terms": {
            "intact": 0,
            "sprain": 2,
            "complete_tear": 4,
        },
        "notes": "sprain covers grade 1-2 / partial injury; complete_tear = grade 3",
    },
    "mm": {
        "finding": "medial meniscus",
        "terms": {
            "intact": 0,
            "degenerative_signal": 1,
            "postoperative": 2,
            "tear": 3,
            "macerated": 4,
        },
        "notes": ("degenerative_signal = intrasubstance signal WITHOUT a morphologic tear; "
                  "tear = any morphologic tear (horizontal, flap, radial, complex); "
                  "postoperative = partial meniscectomy or repair"),
    },
    "lm": {
        "finding": "lateral meniscus",
        "terms": {
            "intact": 0,
            "degenerative_signal": 1,
            "postoperative": 2,
            "tear": 3,
            "macerated": 4,
        },
        "notes": ("degenerative_signal = intrasubstance signal WITHOUT a morphologic tear; "
                  "tear = any morphologic tear (horizontal, flap, radial, complex); "
                  "postoperative = partial meniscectomy or repair"),
    },
    "moa": {
        "finding": "medial compartment osteoarthritis / cartilage loss",
        "terms": {"none": 0, "mild": 1, "moderate": 2, "severe": 3},
        "notes": "grade the described cartilage loss / osteoarthritic change",
    },
    "loa": {
        "finding": "lateral compartment osteoarthritis / cartilage loss",
        "terms": {"none": 0, "mild": 1, "moderate": 2, "severe": 3},
        "notes": "grade the described cartilage loss / osteoarthritic change",
    },
    "pfoa": {
        "finding": "patellofemoral osteoarthritis / cartilage loss",
        "terms": {"none": 0, "mild": 1, "moderate": 2, "severe": 3},
        "notes": "grade the described cartilage loss / osteoarthritic change",
    },
    "effusion": {
        "finding": "joint effusion (fluid)",
        "terms": {"none": 0, "trace_small": 1, "moderate": 2, "large": 3},
        "notes": "'trace' or 'minimal' fluid counts as trace_small (present)",
    },
    "synovitis": {
        "finding": "synovitis / synovial proliferation",
        "terms": {"absent": 0, "present": 3},
        "notes": "only when synovial thickening/proliferation is explicitly described",
    },
    "bakers": {
        "finding": "Baker's (popliteal) cyst",
        "terms": {"absent": 0, "present": 2, "ruptured": 3},
        "notes": "",
    },
    "contusion": {
        "finding": "bone contusion / traumatic bone marrow edema",
        "terms": {"absent": 0, "present": 3},
        "notes": ("TRAUMATIC pattern only; degenerative/reactive marrow changes "
                  "(e.g. due to OA) count as absent"),
    },
    "fracture": {
        "finding": "fracture",
        "terms": {"absent": 0, "present": 4},
        "notes": "includes avulsion fractures and insufficiency/stress fractures",
    },
}

# Formatting-only aliases applied AFTER lower/strip/underscore normalization.
# Only unambiguous radiology synonyms; anything else is a model error.
ALIASES = {
    "acl": {
        "full_tear": "complete_tear", "full_thickness_tear": "complete_tear",
        "rupture": "complete_tear", "ruptured": "complete_tear",
        "partial_thickness_tear": "partial_tear",
        "graft": "reconstructed", "acl_graft": "reconstructed",
        "normal": "intact", "no_abnormality": "intact",
    },
    "mcl": {
        "rupture": "complete_tear", "ruptured": "complete_tear",
        "grade_3": "complete_tear", "grade_1": "sprain", "grade_2": "sprain",
        "normal": "intact",
    },
    "mm": {"rupture": "tear", "torn": "tear", "normal": "intact"},
    "lm": {"rupture": "tear", "torn": "tear", "normal": "intact"},
    "moa": {"normal": "none", "no_oa": "none"},
    "loa": {"normal": "none", "no_oa": "none"},
    "pfoa": {"normal": "none", "no_oa": "none"},
    "effusion": {"trace": "trace_small", "minimal": "trace_small", "small": "trace_small",
                 "no_effusion": "none"},
    "synovitis": {"no_synovitis": "absent"},
    "bakers": {"no_cyst": "absent", "cyst": "present"},
    "contusion": {"bone_marrow_edema": "present", "bme": "present"},
    "fracture": {"avulsion": "present", "stress_fracture": "present",
                 "insufficiency_fracture": "present"},
}
GLOBAL_ALIASES = {"unmentioned": "not_mentioned", "not_stated": "not_mentioned",
                  "n_a": "not_mentioned", "na": "not_mentioned"}


def build_user_content(report: str) -> str:
    lines = []
    for k in KEYS:
        v = VOCAB[k]
        terms = " | ".join(v["terms"].keys())
        lines.append(f"- {k} ({v['finding']}): {terms} | not_mentioned")
        if v["notes"]:
            lines.append(f"  ({v['notes']})")
    vocab_block = "\n".join(lines)
    return f"""You are a musculoskeletal radiology reader. Read the knee MRI report below. It may be written in Spanish, French, English, German, or another language — read it regardless of language.

For each of the 12 findings, choose the SINGLE allowed term that best matches what the report STATES. Rules:
- If the report does not mention the finding at all, use "not_mentioned".
- Do NOT infer findings that are not stated.
- Negated findings ("no tear", "sin rotura", "pas de déchirure", "kein Riss") map to the intact/absent/none term.
- Uncertain/hedged language ("possible", "cannot exclude", "suggestive of") still maps to the matching positive term — choose the term, not the hedge.

Findings and allowed terms:
{vocab_block}

Return ONLY a JSON object with exactly these 12 keys, in this order:
{json.dumps(KEYS)}
Example: {{"acl": "intact", "mcl": "not_mentioned", ...}}
No explanations. No markdown. No extra text.

Report:
\"\"\"{report}\"\"\""""


def build_prompt(report: str) -> str:
    return build_user_content(report)


def _strip_think(text: str) -> str:
    # Defensive: Qwen3 thinking traces must never reach the JSON parser.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def extract_json(text: str):
    """Return parsed dict or None. Finds the first {...} span."""
    text = _strip_think(text)
    m = re.search(r"\{", text)
    if not m:
        return None
    # balanced-brace scan from first '{'
    depth = 0
    start = m.start()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _canon_term(key: str, raw: str):
    t = raw.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    t = re.sub(r"_+", "_", t).strip("_")
    if t in GLOBAL_ALIASES:
        t = GLOBAL_ALIASES[t]
    if t in VOCAB[key]["terms"] or t == "not_mentioned":
        return t
    alias = ALIASES.get(key, {}).get(t)
    if alias:
        return alias
    return None


def normalize_descriptors(raw: dict):
    """-> (terms dict key->term|'not_mentioned', issues list)."""
    terms, issues = {}, []
    for k in KEYS:
        if k not in raw:
            terms[k] = "not_mentioned"
            issues.append(f"missing key: {k}")
            continue
        v = raw[k]
        if not isinstance(v, str):
            terms[k] = "not_mentioned"
            issues.append(f"non-string value for {k}: {v!r}")
            continue
        c = _canon_term(k, v)
        if c is None:
            terms[k] = "not_mentioned"
            issues.append(f"unknown term for {k}: {v!r}")
        else:
            terms[k] = c
    extra = [k for k in raw.keys() if k not in KEYS]
    if extra:
        issues.append(f"extra keys ignored: {extra}")
    return terms, issues


def rank_of(key: str, term: str):
    """Ordinal rank, or None for not_mentioned."""
    if term == "not_mentioned":
        return None
    return VOCAB[key]["terms"][term]


# ---- isotonic calibration (PAVA), no sklearn needed ----

def _pava_fit(xs, ys):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [float(ys[i]) for i in order]
    sums, counts, xmins = [], [], []
    for x, y in zip(xs, ys):
        sums.append(y)
        counts.append(1)
        xmins.append(x)
        while (len(sums) >= 2
               and sums[-2] / counts[-2] > sums[-1] / counts[-1] + 1e-12):
            sums[-2] += sums[-1]
            counts[-2] += counts[-1]
            sums.pop()
            counts.pop()
            xmins.pop()
    # merge blocks sharing the same x (isotonic constraint is vacuous there)
    merged = []
    for xm, s, c in zip(xmins, sums, counts):
        if merged and merged[-1][0] == xm:
            px, ps, pc = merged[-1]
            merged[-1] = (px, ps + s, pc + c)
        else:
            merged.append((xm, s, c))
    return [(xm, s / c) for xm, s, c in merged]


def _pava_predict(levels, x):
    v = levels[0][1]
    for xmin, val in levels:
        if x >= xmin:
            v = val
        else:
            break
    return v


def calibrate_label(ranks, golds, prior):
    """Fit per-label calibration.

    ranks: list of float|None (None = not_mentioned); golds: 0/1 ints.
    Returns dict(prior=..., levels=[(rank, score)...] or None, n_fit=...).
    """
    pairs = [(r, g) for r, g in zip(ranks, golds) if r is not None]
    if len(pairs) < 4 or len({g for _, g in pairs}) < 2:
        return {"prior": prior, "levels": None, "n_fit": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    levels = _pava_fit(xs, ys)
    return {"prior": prior, "levels": levels, "n_fit": len(pairs)}


def apply_calibration(key, term, cal):
    if term == "not_mentioned":
        return cal["prior"]
    r = rank_of(key, term)
    if cal["levels"] is None:
        # fallback: linear in rank fraction (only when too few gold mentions)
        maxr = max(VOCAB[key]["terms"].values())
        return 0.05 + 0.90 * (r / maxr)
    return float(_pava_predict(cal["levels"], r))


def rank_score(key, term):
    """Uncalibrated ordinal score in [0,1] — same AUC as calibrated."""
    if term == "not_mentioned":
        return 0.0
    maxr = max(VOCAB[key]["terms"].values())
    return rank_of(key, term) / maxr


# ---- metrics ----

def auc_score(y_true, y_score):
    """Binary AUC via Mann-Whitney. Returns None if degenerate."""
    y_true = list(y_true)
    y_score = list(y_score)
    n1 = sum(y_true)
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s = sum(ranks[i] for i in range(len(y_true)) if y_true[i] == 1)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def macro_auc(gold: dict, scores: dict):
    """gold/scores: label -> list. Returns (macro, {label: (auc, n_pos, n)})."""
    per = {}
    vals = []
    for lab in gold:
        a = auc_score(gold[lab], scores[lab])
        n1 = sum(gold[lab])
        per[lab] = (a, n1, len(gold[lab]))
        if a is not None:
            vals.append(a)
    macro = sum(vals) / len(vals) if vals else None
    return macro, per
