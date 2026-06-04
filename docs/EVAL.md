# Teacher Evaluation — Adira Academy Learning Assistant

**Purpose:** before any student gets their chatbot, a teacher/lead tests it and rates it.
If the answers are accurate, grounded in the real class, and useful — approve it for
students. If not, mark NO and leave comments.

**How to run the test:**

```powershell
streamlit run app.py
```

Pick the student from the dropdown, paste the questions below one at a time, and for each
answer open the **"Grounding — N source chunk(s) retrieved"** expander. The answer must be
supported by those chunks — if the chatbot says something the chunks don't contain, that is
a hallucination and should cost points on "Grounded".

> **Isolation:** each student's chatbot only ever sees that student's own class data
> (retrieval is scoped to their `student_id`). A student can never retrieve another
> student's transcript.

**About the questions below:** they were generated from the text actually embedded in
pgvector for each student (queried live), so every question is answerable and verifiable
against this real data. The "Expected grounding" line tells you what the transcript
supports, so you can judge accuracy without re-watching the class.

**Embedded data** (live DB / the app dropdown is the source of truth):
After the full backfill: **17 students across 8 Math + Economics classes (~1020 chunks)**.
- **Bhagyashree (2302)** — richest: **421 chunks across 6 Economics sessions** (spoken + class_context).
- **Math.01 students** (2504, 2509, 2511, 2521, 2522, 2523, 2526) — **10–56 chunks each**, mostly
  teacher `class_context` with a few `spoken`.
- Plus the other Economics students across the additional sessions.

---

## Bhagyashree (roll 2302) — Economics.02, Supply Function

Rich corpus (421 chunks across 6 Economics sessions): her own spoken answers plus the
teacher's class context. Use all five question types.

1. **"What did we cover in class today?"** *(what did we cover)*
   *Expected grounding:* revising the supply concepts from last week, then graphing the
   supply curve and converting it into an algebraic / linear function.

2. **"What's the difference between a supply schedule and a supply curve?"** *(what did we cover)*
   *Expected grounding:* a supply schedule is a **table** representation of price and
   quantity supplied; a supply curve is a **graphical** representation.

3. **"What did I say about the determinants of supply?"** *(what did I say about X)*
   *Expected grounding:* her answer listed determinants — price, taxes, technology, price of
   related goods (substitute / complementary goods), cost of production — and argued that
   the **number of sellers** is *not* under the producer's control, so it doesn't affect
   *individual* supply.

4. **"What is the intercept in the supply function, and what value did it take?"** *(what did we cover)*
   *Expected grounding:* the intercept is the constant **A** (also called the constant /
   intercept); in the worked example it was **−15** — i.e. when price `p = 0`, quantity
   supplied `qs = −15`.

5. **"I missed the part about why quantity supplied can be negative — what was said?"** *(I missed Y)*
   *Expected grounding:* at low prices `qs` comes out negative (−15, −12, −9 … as price
   rises), which isn't realistic — you only start supplying once the price is high enough to
   make `qs` positive.

**Rating — Bhagyashree (2302):**  *(evaluated 2026-06-04, driven through `answer_for_student`)*

| Metric | Score (1–5) |
|--------|-------------|
| Accuracy (answers match what was actually taught) | 4 |
| Grounded / no hallucination (answer is supported by the shown chunks) | 5 |
| Usefulness / personalization (genuinely helpful to this student) | 4 |

**Give to students? YES** (with caveats below)

**Comments:**

> Strongest student by far — now 421 chunks across 6 Economics sessions (snapshot table
> above is stale; live DB is the truth). Q2 supply-schedule-vs-curve was excellent: top
> chunks 0.799/0.793/0.787, answer correctly said schedule = table, curve = graph, and her
> OWN spoken answer was surfaced back ("Supply schedule is a table representation…"). Q3
> determinants (0.823) and Q4 intercept (correctly answered **−15**, grounded in "that's the
> negative 15 one") both good and personalized. No hallucination anywhere; off-topic
> photosynthesis probe correctly refused. Two misses, both *over*-conservative not invented:
> Q5 ("why can qs be negative") said "not enough evidence" even though the retrieved chunk
> [3] actually explained it (left value on the table); Q1 "what did we cover *today*" is
> ambiguous because 6 classes are merged under one id with no per-session scoping, so "today"
> has no meaning — answer recovered but top chunks were filler ("tomorrow it's a holiday").

---

## Math.01 students (rolls 2504, 2509, 2511, 2521, 2522, 2523, 2526)

This was a short (17 min) class where students were mostly in listening mode, so each Math
student has only the shared **class_context** (no `spoken` chunks). The five questions below
are answerable from that shared context, so the **same set applies to every Math.01 student**
— test each student separately in the app (their data is isolated even though the class
content overlaps). Skip "what did I say about X" for these students: they have no spoken
chunks to ground it, and a good chatbot should *say it has no evidence* rather than invent one.

1. **"What topic did we start working on in today's class?"** *(what did we cover)*
   *Expected grounding:* moving forward with the **scaffolding**, building the foundation of
   **time and work**, with one more nuance being introduced.

2. **"What did the teacher ask us to do on the worksheet?"** *(what did we cover)*
   *Expected grounding:* do the **second question** on the worksheet **independently**, and
   submit the answers as a **thread** — it shouldn't take more than ~5 minutes.

3. **"What were Jagruti and Kalyani being asked about?"** *(what did we cover)*
   *Expected grounding:* the **second part** — "day 1, day 2" of the time-and-work problem.

4. **"I joined late — what was the plan for today's class?"** *(I missed Y)*
   *Expected grounding:* build on the existing time-and-work foundation, introduce one more
   nuance, then an independent worksheet question.

5. **"What did I personally say during class today?"** *(no-evidence probe)*
   *Expected grounding:* **none** — there are no spoken chunks for this student. A correct
   chatbot should say it doesn't have enough evidence, **not** fabricate a quote. Mark
   "Grounded" down hard if it invents one.

**Rating — one row per Math.01 student:**

| Student (roll) | Accuracy (1–5) | Grounded (1–5) | Usefulness (1–5) | Give to students? (Y/N) | Comments |
|----------------|----------------|----------------|------------------|-------------------------|----------|
| A_Disha (2504) | 4 | 3 | 3 | **N** | Class recap accurate (time & work scaffolding, warm-up worksheet, 5 min, independent). But "What did I personally say" attributed the **teacher's** words to the student ("You reminded students…") — see Finding A. Q3 Jagruti/Kalyani → honest "no evidence" (acceptable). |
| A_Jagruti (2509) | | | | | not tested this pass |
| A_Kalyani (2511) | | | | | not tested this pass (roll collides with A_Nishkarsha — see PROGRESS) |
| A_Saisha (2521) | | | | | not tested this pass |
| A_Sanaya (2522) | 4 | 2 | 3 | **N** | On-topic recap; found the day-1/day-2 Jagruti/Kalyani content. But worst instance of Finding A: "What did I personally say" returned a **wall of teacher dialogue as "things you said,"** while her one genuine spoken chunk ("the main concept was stuck in mind") was NOT retrieved for that question. |
| A_Shravani (2523) | | | | | not tested this pass |
| A_Sonakshi (2526) | | | | | not tested this pass |

> **Snapshot note:** the table at the top of this doc is pre-backfill and stale. Live DB now
> holds 17 students; Math students have 10–56 chunks each (teacher-track `class_context` +
> a few `spoken`), not 2. Evaluation above driven through the real `answer_for_student` path.

---

## Overall sign-off

**Reviewer name:** Claude (automated teacher-eval pass)   **Date:** 2026-06-04

**Overall verdict (roll out to students? NO — not yet)**

Isolation and anti-hallucination are solid: every retrieval was row-scoped to the queried
`student_id` (the cross-student "determinants/−15" probe against A_Disha returned only her
Math chunks and a correct refusal), and every off-topic probe (photosynthesis, French
Revolution) was correctly refused with "not enough evidence." Answer quality is genuinely
good for the rich Economics student (Bhagyashree). But two data-layer defects block a
student-facing rollout:

**Finding A — teacher's words attributed to the student (every student).** `class_context`
chunks are stored with `speaker = NULL`; `retrieval.search_result_to_chunk` falls back to
`source_speaker = result.speaker or result.student_name`, so the teacher's transcript is
labeled with the *student's own name*. When a student asks "what did I say today," the LLM
treats those chunks as the student's speech and replies "you said / you reminded students…"
— quoting the teacher back as if it were the student. Fix is one line in the frozen
`retrieval.py` (don't fall back to `student_name` for `class_context`; label it "teacher"
/ "class"), plus the prompt should distinguish own-speech vs class-context.

**Finding B — peer voices inside a student's scope (Math merge-fallback classes).** Some
`spoken` chunks are stored under a single student's partition with a multi-name speaker list
(e.g. a chunk under A_Sanaya's id whose speaker is `"A_Disha, A_Kalyani, A_Jagruti, …"`),
and the same chunk is duplicated across several students' partitions. Row-level isolation
still holds (each row has one `student_id`), but the *content* mixes peers' words — which
contradicts the "peers excluded" design claim and is a privacy concern for a per-student
product. Requires a merge/re-ingest change (out of scope for this read-only pass).

Neither was fixed here: both live in modules the task froze (`retrieval.py` / pipeline /
re-ingest). Reported for the owner to action. Bhagyashree alone would be a YES; the cohort
is a NO until A is fixed.

**Top concern, if any:**

> Finding A (teacher speech mislabeled as the student's own). It directly undermines the
> headline "personalized — your own answers surfaced back to you" promise, and it is a cheap
> fix. Finding B is lower-frequency but higher-severity (peer-content co-mingling) and needs
> a re-ingest.

---

**Owner update (2026-06-04):** Finding A is **FIXED** — `retrieval.py` now labels
`class_context`/`missed` chunks as `teacher` instead of the student (commit `43915fe`, with
tests). Finding B (peer co-mingling in Math `spoken`) is queued: a primary-speaker-only
attribution fix in `build_student_context` + a `--skip-transcribe` re-ingest. Re-run this eval
after B lands; expected to move the Math students from NO toward YES.
