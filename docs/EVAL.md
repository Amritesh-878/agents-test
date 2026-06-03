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

**Snapshot of embedded data at time of writing** (more classes get added later — the live
DB is the source of truth):

| Student | Roll | Class | Chunks | Chunk types present |
|---------|------|-------|--------|---------------------|
| Bhagyashree | 2302 | Economics.02 — Supply Function (16 Apr, 51 min) | 192 | spoken, class_context |
| A_Disha | 2504 | Math.01 — Time & Work (08 Apr, 17 min) | 2 | class_context |
| A_Jagruti | 2509 | Math.01 — Time & Work | 2 | class_context |
| A_Kalyani | 2511 | Math.01 — Time & Work | 2 | class_context |
| A_Saisha | 2521 | Math.01 — Time & Work | 2 | class_context |
| A_Sanaya | 2522 | Math.01 — Time & Work | 2 | class_context |
| A_Shravani | 2523 | Math.01 — Time & Work | 2 | class_context |
| A_Sonakshi | 2526 | Math.01 — Time & Work | 2 | class_context |

---

## Bhagyashree (roll 2302) — Economics.02, Supply Function

Rich corpus (192 chunks): her own spoken answers plus the teacher's class context. Use all
five question types.

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

**Rating — Bhagyashree (2302):**

| Metric | Score (1–5) |
|--------|-------------|
| Accuracy (answers match what was actually taught) | |
| Grounded / no hallucination (answer is supported by the shown chunks) | |
| Usefulness / personalization (genuinely helpful to this student) | |

**Give to students? YES / NO:**

**Comments:**

> _____________________________________________________________________

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
| A_Disha (2504) | | | | | |
| A_Jagruti (2509) | | | | | |
| A_Kalyani (2511) | | | | | |
| A_Saisha (2521) | | | | | |
| A_Sanaya (2522) | | | | | |
| A_Shravani (2523) | | | | | |
| A_Sonakshi (2526) | | | | | |

---

## Overall sign-off

**Reviewer name:** ________________________   **Date:** ____________

**Overall verdict (roll out to students? YES / NO):**

**Top concern, if any:**

> _____________________________________________________________________
