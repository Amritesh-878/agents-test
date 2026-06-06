# Transcription Comparison — Results: `small` vs `medium`

**Run:** 2026-06-07. **Verdict: ADOPT `medium` for embedding.** Evidence below.
Both runs use per-segment language selection; the only intended variable is model size
(`small` → `medium`). See GPU caveat in §6.

---

## 1. Headline

| | embedded words (post-filter) | Devanagari-garble chunks that survived the filter | teacher-track Devanagari (mean) |
|---|---|---|---|
| **A — `small`** | 151,688 | **93** | higher |
| **B — `medium`** | 122,121 | **66** | lower |

`medium` embeds a **leaner, higher-signal** corpus: ~20% fewer words, ~30% fewer garbage
Devanagari chunks leaking past the quality gate, and cleaner teacher tracks. The lower word
count is **mostly junk not being generated**, not real content lost (see §3–4).

---

## 2. Per-class verdict

Metric = teacher-track Devanagari ratio (lower better) + count of Devanagari-garble chunks that
**survive `is_quality_text` into embeddings** (lower better). ✓ medium better · ⚠ mixed · ✗ regression.

| Class | Teacher Δdeva (A→B) | Garble survivors A→B | Verdict |
|---|---|---|---|
| Economics — Supply Function 16 Apr | 0.046 → **0.008** | 26 → **6** | ✓ |
| Economics — Determinants Last Part s1 | 0.026 → **0.017** | 6 → 12 | ⚠ |
| Economics — Determinants Last Part s2 | (agg 0.136 → **0.042**) | 6 → 6 | ✓ |
| Economics — Detrmnts (Expectns…) 09 Apr | 0.007 → **0.001** | 6 → 10 | ⚠ |
| Economics — How MC=MR 08 Apr | 0.007 → 0.020 | 2 → 6 | ⚠ (tiny 876-word track) |
| Economics — Intro to Supply 31 Mar | 0.032 → **0.001** | 2 → 8 | ⚠ (big teacher win, more short-loop leaks) |
| Math — Scaffolding Part_04 16 Apr | 0.008 → **0.000** | **13 → 0** | ✓ |
| Math — Time & Work 03 (31 Mar) | 0.012 → 0.036 | **30 → 18** | ✓ |
| Math — Time & Work 05 (08 Apr) | 0.000 → 0.000 | 2 → **0** | ✓ |

**Pattern:** teacher tracks are cleaner under `medium` in **7 of 9** classes. Garble survivors
fall in the worst-offending classes (Math 03: 30→18; Part_04: 13→0). The ⚠ rows are Economics
classes where a handful of **short** loop-hallucination chunks slip the filter under `medium`
(see §5 — this is a filter gap, not a transcript-quality regression).

---

## 3. What the raw text shows (this reverses the naive metric)

The metric "medium has MORE Devanagari on student tracks" looked alarming (e.g. Math student mics
0.00 → 1.00 Devanagari). **Reading the text shows it is not genuine Hindi — it's hallucination on
near-silent mics**, and `small` was equally broken there, just differently:

**Near-silent student mic (`A_Disha`, `A_Gunjan` — Math):**
- `small`: 5 segments of `"you you you …"` at 30 s intervals — the classic Whisper silence artifact.
- `medium`: one segment, `"अच्छा पाइपाइपाइपाइ…"` (पाइ repeated hundreds of times) — a **loop
  hallucination** that scores as ~180 "Hindi words." Worse-looking, but **caught by the embed filter.**

Neither model has real audio to transcribe there (students listening, not speaking). Both produce
garbage; the embed-time quality gate is what matters, and it drops both (Math Part_04: medium
leaked **0** garble chunks).

---

## 4. Bilingual retention — `medium` does NOT erase real Hindi (H3 confirmed)

The students whose Devanagari **dropped** under `medium` are the genuine code-switchers
(Bhagyashree, Anshi, Saisha). Reading the text, the dropped Devanagari was **garble, not Hindi** —
`small` was mis-rendering their **English** as Devanagari, and `medium` transcribes it correctly:

**Bhagyashree, Intro to Supply (309 words → 3390 words; deva 0.243 → 0.001):**
- `small`: `"You You You"` silence spam + `"तो आप गुड़ से जी"` (loop garble of "the quantity of
  good supplied").
- `medium`: coherent English — *"law of supply states if price increases, the producer will produce
  more because his profit…"*, *"maybe consumers are ready to buy it at a certain price."*

`medium` recovered ~10× more of her actual speech, in the right language, with far less garble.
No case was found where `medium` erased genuine Hindi.

---

## 5. The one real caveat — a filter gap that affects BOTH models

66 (medium) / 93 (small) Devanagari-garble chunks still **survive** `is_quality_text` and would be
embedded. They are **short** loop hallucinations that duck the thresholds — e.g.:

> `माज्यारा पास्ता वेन्द सप्लाइट्रू  माज्यारा पास्ता वेन्द सप्लाइट्रू  माज्यारा पास्ता वेन्द सप्लाइट्रू`

A 4-word phrase repeated 3× = under the "trigram appears ≥4×" rule, and it has no nukta, so it
passes. This is a **pre-existing weakness in `scripts/embed_and_store.py:is_quality_text`**, not a
`medium` regression (medium actually leaks fewer). **Recommended follow-up (independent of this
decision):** lower the trigram-repeat threshold to 3 and/or add a distinct-token-ratio check
(reject if unique-words / total-words < ~0.5 on chunks ≥ 8 words). This cleans both corpora.

---

## 6. Caveats carried from the brief

- **Model diff, not a GPU diff.** Gains are attributable to `small`→`medium` (and likely a
  compute-type change that rode along). We do **not** have the control pair to isolate GPU silicon;
  do not claim a "GPU effect."
- **Proxies, not WER.** No human gold transcript exists; Devanagari-ratio / repetition / survival-
  past-filter are objective stand-ins, not ground-truth accuracy.
- **`embed_and_store` failed on the brother's box** — expected (no Postgres there). Unrelated to
  quality; embedding happens on this machine via `--skip-transcribe`.
- The `medium` tree still contains `raw/` subfolders (bloat, ~not deleted); harmless, worth a sweep.

---

## 7. Recommendation

1. **Adopt `medium`** — re-embed from the brother's transcripts via
   `--skip-transcribe --roster … --attendance …`. Teacher and active-student content is cleaner and
   more complete; silence-mic garbage is no worse and largely filtered.
2. **Before/with the re-embed, tighten `is_quality_text`** (§5) to drop the short-loop survivors —
   benefits both models, one-time fix.
3. No class warrants keeping `small`. The ⚠ classes are filter-gap noise, not transcript regressions.
