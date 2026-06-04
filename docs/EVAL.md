# Teacher Evaluation — Adira Academy Learning Assistant

Please help us check each student's chatbot **before** students get to use it. About 15 minutes.

---

## How to test — 3 steps

1. **Open it.** A team member starts it with `streamlit run app.py`; it opens in your browser.
2. **Pick a student** from the dropdown at the top of the page.
3. **Ask that student's questions** (listed below), one at a time. After each answer, click the
   **"Grounding"** box to see the class transcript the bot used to answer.

## What you're grading

For each student, score these **1 to 5** (5 = best):

- **Accuracy** — does the answer match what was actually taught?
- **Grounded** — is the answer backed by the transcript shown in the "Grounding" box (not made up)?
- **Useful** — would this genuinely help the student?

Then mark **Give to students? Yes / No**.

> If the bot says something the transcript doesn't show, that's a made-up answer — score
> **Grounded** low. If the bot honestly says "I don't have enough evidence" when a student
> barely spoke, that's **correct** — don't punish it.

---

## Student 1 — Bhagyashree (Economics)

**Ask:**
1. What did we cover in class?
2. What's the difference between a supply schedule and a supply curve?
3. What did I say about the determinants of supply?
4. What is the intercept in the supply function, and what value did it take?
5. I missed the part about why quantity supplied can be negative — what was said?

*Answer key (what the class actually covered):* supply schedule = a **table**, supply curve = a
**graph**; intercept is the constant **A**, value **−15**; quantity supplied is negative at low
prices, which isn't realistic until price rises enough.

**Your grades:**

| Measure | Score (1–5) |
|---------|:-----------:|
| Accuracy |  |
| Grounded |  |
| Useful |  |

**Give to students?**  ☐ Yes   ☐ No

**Comments:**

> _______________________________________________________________

---

## Students 2–4 — Math (test any 2–3 of these)

Pick from: **A_Disha (2504), A_Jagruti (2509), A_Kalyani (2511), A_Saisha (2521),
A_Sanaya (2522), A_Shravani (2523), A_Sonakshi (2526)**. The same questions work for every Math
student — their data is kept separate even though the class is shared.

**Ask:**
1. What topic did we start working on in class?
2. What did the teacher ask us to do on the worksheet?
3. What were Jagruti and Kalyani being asked about?
4. I joined late — what was the plan for class?
5. What did I personally say during class?  *(Many Math students barely spoke — if the bot says
   "not enough evidence," that's the correct answer, not a failure.)*

*Answer key:* started the **time-and-work scaffolding**; do the **2nd worksheet question
independently** (~5 min, submit as a thread); Jagruti & Kalyani were asked about **day 1 / day 2**
of the problem.

Fill one box per student you test:

**Student: ________________  (roll ______)**

| Measure | Score (1–5) |
|---------|:-----------:|
| Accuracy |  |
| Grounded |  |
| Useful |  |

**Give to students?**  ☐ Yes   ☐ No   **Comments:** ___________________________________

---

**Student: ________________  (roll ______)**

| Measure | Score (1–5) |
|---------|:-----------:|
| Accuracy |  |
| Grounded |  |
| Useful |  |

**Give to students?**  ☐ Yes   ☐ No   **Comments:** ___________________________________

---

**Student: ________________  (roll ______)**

| Measure | Score (1–5) |
|---------|:-----------:|
| Accuracy |  |
| Grounded |  |
| Useful |  |

**Give to students?**  ☐ Yes   ☐ No   **Comments:** ___________________________________

---

## Overall sign-off

**Ready to give to students?**   ☐ Yes   ☐ No

**Biggest concern (if any):**

> _______________________________________________________________

**Reviewer name:** ____________________     **Date:** ______________

---
---

## Appendix — automated pre-eval (for the team, not teachers)

An automated pass on 2026-06-04 found two defects, **both since fixed** — re-run this eval to
confirm before the teacher session:

- **Finding A** — the teacher's words were being quoted back as the student's own on "what did I
  say" questions. Fixed: retrieval now labels teacher chunks as `teacher` and scopes
  self-referential questions to the student's own `spoken` chunks (commits `43915fe`, `09dd062`).
- **Finding B** — a student's `spoken` could contain a peer's words on overlapping speech. Fixed:
  segments are attributed to the primary speaker only, and the store was re-ingested (`f8e687d`).

Isolation (each student sees only their own data) and refusal-on-no-evidence both passed and were
unaffected. Known limitation (not a bug): some Math students' own speech is sparse/garbled due to
the small transcription model — the transcription-quality upgrade is tracked separately.
