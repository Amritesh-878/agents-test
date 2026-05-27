# Master Implementation Plan

**Project:** [Project Name]

**Date Created:** [Date]

---

## 🎉 PROJECT COMPLETION SUMMARY

**Status:** 🔄 **IN PROGRESS** ([Date])

**Overview of all tasks:**

| Phase | Task        | Status | Build | Tests |
| ----- | ----------- | ------ | ----- | ----- |
| [#]   | [Task Name] | 🔄     | ❓    | ❓    |
| [#]   | [Task Name] | ⏳     | ❌    | ❓    |

**Current Verification ([Date]):**

- 🔄 Build: [Status]
- 🔄 Tests: [X/Y passing]
- 🔄 Integration: [X/Y passing]
- 🔄 Formatting: [Status]

**Deliverables:**

- [Primary deliverable 1]
- [Primary deliverable 2]
- [Primary deliverable 3]

---

## Table of Contents

1. [Implementation Order](#implementation-order)
2. [Dependency Graph](#dependency-graph)
3. [Task Status Tracker](#task-status-tracker)
4. [Phase Summaries](#phase-summaries)
5. [Handoff Notes](#handoff-notes)
6. [Critical Dependencies](#critical-dependencies)

---

## Implementation Order

### Rationale

**Why this order?**

---

## Phase [#]: [Phase Name]

### [#️⃣] TODO [#]: [Task Name]

**Why [this position]:**

- [Reason 1]
- [Reason 2]
- [Reason 3]

**Scope:**

- [Change 1]
- [Change 2]
- [~X files affected]

**Success Criteria:**

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

---

### [#️⃣] TODO [#]: [Task Name]

**Why [this position]:**

- [Reason 1]
- [Reason 2]

**Scope:**

- [Change 1]
- [Change 2]
- [~X files affected]

**Success Criteria:**

- [ ] [Criterion 1]
- [ ] [Criterion 2]

---

## Dependency Graph

**OPTIONAL** - Include if multiple tasks have complex dependencies

```
┌─────────────────────────────┐
│  TODO [#]: [Task Name]      │  ⬅️ START HERE
└─────────────────────────────┘
           │
           ├─────────┬─────────┐
           ▼         ▼         ▼
    ┌───────────┐ ┌───────┐ ┌────────┐
    │ TODO [#]  │ │TODO[#]│ │TODO[#] │
    │ [Task]    │ │[Task] │ │[Task]  │
    │ (time)    │ │(time) │ │(time)  │
    └───────────┘ └───────┘ └────────┘
```

---

## Task Status Tracker

**Update this table as each task is completed. Append notes below each completed task.**

| Phase | TODO | Title       | Status         | Notes                                     |
| ----- | ---- | ----------- | -------------- | ----------------------------------------- |
| [#]   | [#]  | [Task Name] | ⏳ Not Started | See [TODO [#] Handoff](#todo-[#]-handoff) |
| [#]   | [#]  | [Task Name] | 🔄 In Progress | See [TODO [#] Handoff](#todo-[#]-handoff) |

**Status Legend:**

- 🔄 In Progress
- ⏳ Blocked / Waiting
- ✅ Completed
- ❌ Failed / Needs Rework

---

## Phase Summaries

### Phase [#]: [Phase Name]

**Phase Goal:** [Overall purpose]

**What gets built:**

- [System/feature 1]
- [System/feature 2]
- [System/feature 3]

**What gets deleted:**

- [Old file/system 1]
- [Old file/system 2]

**What stays the same:**

- [Important: what doesn't change]

**Impact:** [How does this improve the codebase?]

---

### Phase [#]: [Phase Name]

**Phase Goal:** [Overall purpose]

**What gets built:**

- [System/feature 1]

**What gets deleted:**

- [Old file/system 1]

**Impact:** [How does this improve the codebase?]

---

## Handoff Notes

**Instructions for implementors:** After completing a TODO, update this section with blockers, decisions, and important information for the next person/agent.

### TODO [#] Handoff

**Status:** 🔄 In Progress (or ✅ Completed)

**Prerequisites met:**

- [x] Prerequisite 1
- [x] Prerequisite 2

```
Completed by: [Name/AI Model]
Build status: ✅ Passing (or 🔄 In Progress)

### What was done:
- [Change 1 with impact]
- [Change 2 with impact]
- [Change 3 with impact]

### Blockers encountered:
- [Blocker 1 and resolution]
- [Or: None]

### Breaking changes:
- [Breaking change 1 - migration path]
- [Breaking change 2 - migration path]

### Files modified count: [X]
### Files created count: [X]
### Files deleted count: [X]

### Key decisions made:
- [Design decision 1 - rationale]
- [Design decision 2 - rationale]

### Tests passing: ✅ All ([X] tests) (or 🔄 [X/Y passing])

### Warnings to next implementor (TODO [next]):
- [Important warning 1]
- [Important warning 2]

### Information for TODO [next] implementor:
- [API change needed]
- [File location change]
- [Updated terminology]

### Critical notes:
- [Any critical information for handoff]
```

---

### TODO [#] Handoff

**Status:** ⏳ Not Started (or 🔄 In Progress / ✅ Completed)

**Prerequisites from TODO [#]:**

- [ ] [Prerequisite 1]
- [ ] [Prerequisite 2]

```
[When completed, fill in like above example]
```

---

## Critical Dependencies

⚠️ **DO NOT SKIP OR REORDER - These must be sequential:**

| Violation                          | Consequence                          |
| ---------------------------------- | ------------------------------------ |
| Implement TODO [X] before TODO [Y] | [System Z would be broken - explain] |
| [Violation 2]                      | [Consequence]                        |

---

## Parallel Work (OPTIONAL)

You may parallelize these to save time:

**During TODO [X]:** [Task that can happen in parallel]
**During TODO [Y]:** [Task that can happen in parallel]

---

## How to Update This Document

**After completing each task:**

1. Update the status table (Status column)
2. Fill in the handoff section for that TODO
3. Update the next task's prerequisites if any changed
4. Commit with message: `chore: Complete TODO [N] implementation`

**Example commit message:**

```
chore: Complete TODO [#] - [Task Name]

- [What was done 1]
- [What was done 2]
- All tests passing
- Handoff notes added for TODO [Next]
```

---

## Summary

**Key Principle:** [Guiding principle of the implementation order]

Begin with [TODO [#]: Task Name] when ready. See [TASK_PLAN_TEMPLATE.md](./TASK_PLAN_TEMPLATE.md) for individual task details.
