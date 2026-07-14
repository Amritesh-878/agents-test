# Adira Academy Learning Assistant — Teacher Re-test (Round 2)

Thank you for testing round 1! Your feedback was clear: the bot handled *recall* questions ("what did
we cover", "what did I say") well, but it **could not answer concept / "related" questions** — the kind
where you connect ideas ("how do the determinants relate to the supply function?") — and it lost to a
general AI on explanation. The root cause was that its only source was a **noisy transcript of what was
said aloud**.

**What changed:** we now also load each class's **own materials** (the slides / PDF / notes you teach
from) as an *authoritative* source. The bot can now explain and connect concepts **from your material**,
and it labels where each answer came from ("according to the class material…" vs "in class the teacher
said…" vs "you said…"). It still **refuses** when neither the transcript nor the material covers
something — that safety behaviour is unchanged.

This round is short: please re-ask the questions that broke last time, plus a few fresh concept
questions, and tell us whether the answers are now correct and genuinely more useful than before.

---

## Before you start (for the person running it)

- This round only works after the class's materials have been **ingested** into the store:
  `python -m scripts.ingest_materials --materials-dir "materials\<class>" --class-name "<class>" --identity-map "output\<class>\identity_map.json"`.
- If materials are **not** ingested yet, the concept questions below will (correctly) say "not enough
  evidence" — that is the not-yet-loaded state, not the finished behaviour. Confirm ingestion first.
- Everything else (login, student/class dropdowns, the **Grounding** box) works exactly as in round 1.
  The Grounding box now also shows **material** chunks with their **source file** — that is how you see
  the answer came from the slides, not from a general AI.

---

## How to read each question's tag

Each question is tagged so your feedback stays interpretable:

- **[concept / materials]** — the new capability. Should be answered **from your slides/notes** and
  should **attribute** the material. *This is what broke in round 1.*
- **[recall / transcript]** — unchanged from round 1; a regression check that we didn't break recall.
- **[self-referential]** — "what did *I* say?"; must use only the student's own words (never the
  slides). A regression check that material does **not** leak into personal-contribution answers.
- **[refusal — should decline]** — a real topic the class never covered. The bot **should** say it has
  no evidence and **must not** answer from general knowledge. A refusal here is the **correct** answer.

---

## Chatbot 1 — Bhagyashree (Economics)

Ask these one at a time. The ⭐ column marks the ones we most want you to confirm are correct *every
time* (we turn those into permanent automatic tests).

| # | Question | Tag | What a good answer does | ⭐ |
|---|----------|-----|-------------------------|----|
| 1 | **How are the determinants of supply connected to the supply function?** | concept / materials | Explains that the determinants (taxes, technology, price of related goods…) **shift / change** the supply function or curve, **attributed to the class material**. *(This is the exact question that failed in round 1.)* | ☐ |
| 2 | Explain the supply function again in simple words. | concept / materials | A clean explanation grounded in the slides, attributed. | ☐ |
| 3 | Why does quantity supplied come out negative at low prices? | concept / materials + transcript | Connects the supply-schedule numbers to the idea; may draw on both material and what was said. | ☐ |
| 4 | What's the difference between a supply schedule and a supply curve? | recall / transcript | schedule = a **table**, curve = a **graph**. | ☐ |
| 5 | What did I say about the intercept in the supply function? | self-referential | Only **Bhagyashree's own** words (the intercept "should be negative"); must **not** quote the teacher or slides as hers. | ☐ |
| 6 | How is **GDP** calculated, and what are the components of national income? | refusal — should decline | GDP / national income is **not** in these supply classes or their materials. The bot **should decline**, not answer from general knowledge. | ☐ |

*For your reference — what these classes actually covered:* the **determinants of supply** (taxes,
technology, price of related goods / substitutes / complements); the **supply function** (intercept =
the constant **A**, value **−15**); the **supply schedule** (a table) vs the **supply curve** (a graph);
quantity supplied is **negative at low prices** until the price rises enough.

**Your grades:**

| Measure | Round 1 (if you recall) | Round 2 |
|---------|:-----------------------:|:-------:|
| Accuracy (1–5) |  |  |
| Grounded (1–5) |  |  |
| Useful (1–5) |  |  |

**Give to students now?**  ☐ Yes   ☐ No   ☐ Not yet

**Specifically on the concept questions (1–3): better than round 1?**  ☐ Much better   ☐ Somewhat   ☐ No change   ☐ Worse

**If an answer was wrong — why?**  ☐ made-up (not in Grounding)   ☐ incomplete   ☐ wrong person (teacher's/other's words as the student's)   ☐ answered from general knowledge when it should have declined   ☐ too vague

**Comments:**

> ___________________________________________________________________________

---

## Try your own concept questions (free test)

The whole point of this round is concept / "related" questions. Please also ask **2–3 of your own** —
the kind a student would ask when they *understood the words but not the idea*. Some starters:

- "How does a change in technology affect the supply curve?"
- "If taxes go up, what happens to supply and why?"
- "Connect the supply schedule to the supply curve for me."
- "Explain in one line why the intercept is negative."

For each, note whether the answer was **grounded in your material** (check the Grounding box), correct,
and better than what round 1 could do.

| Your question | Grounded in material? (Y/N) | Correct? (Y/N) | Better than round 1? | ⭐ Lock in? | Notes |
|---|:---:|:---:|:---:|:---:|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

---

## Overall sign-off (round 2)

**Did adding the class materials close the "can't answer related questions" gap?**
☐ Yes, clearly   ☐ Partly   ☐ No

**Ready to test with real students?**   ☐ Yes   ☐ No   ☐ Not yet

**Top thing still to fix:**

> ___________________________________________________________________________

**Reviewer name:** ____________________________     **Date:** __________________

---

> **Note (internal):** the automated counterparts of these questions live in `data/eval_qa.json`
> (`bhagyashree-concept-determinants-supply-link` for Q1, `bhagyashree-concept-not-in-source-refusal`
> for Q6, plus the existing self-referential and recall cases). Q1's automated case stays failing until
> this class's materials are ingested — that is the honest checkpoint (TASK-021): if the concept
> questions still fail here after ingestion, the materials did not close the gap and Phase 2 should not
> start.
