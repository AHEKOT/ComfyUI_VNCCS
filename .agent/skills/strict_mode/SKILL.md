---
name: universal-strict-protocol
description: Enforces a context-aware workflow for questions, minor changes, and complex code modifications with mandatory planning and testing.
---

# Universal Protocol: Understand -> Decide -> Plan (if needed) -> Execute -> Verify

This skill enforces a context-sensitive workflow that adapts to the type and clarity of the user’s request.

## Core Logic

**Step 1: Classify Request**
*   **Type A (Info)**: Questions, explanations. -> **Action**: Answer immediately. No Artifacts.
*   **Type B (Simple)**: Explicit, unambiguous changes ("Delete line X"). -> **Action**: Execute immediately. Verify.
*   **Type C (Complex)**: Refactoring, new features, ambiguous requests. -> **Action**: STOP & PLAN.

**Step 2: Execution Path**

**If Type C (Complex):**
1.  **Stop**: Do not touch code.
2.  **Plan**: Create `implementation_plan.md` (Goal, Changes, Verification). Structured, beautifully designed with code examples (original and modified).
3.  **Approve**: Ask user for "Yes/No".
4.  **Execute**: Only after approval.
5.  **internal planning based onVerification Standards**: Make before toching code. Check each step after each phase

**If Type A or B:**
*   Proceed directly. Ensure correctness.
*   **MANDATORY**: After ANY code change (Type B or C), update `CHANGELOG.md` IMMEDIATELY.

## 4. Verification Standards (Mandatory)
*   **Syntax Check**: Must run python syntax check (`python -m py_compile file.py`) or linter.
*   **Runtime Check**: If possible, trigger the code (e.g., import test, dry run).
*   **Evidence**: Output logs or errors must be analyzed in the final report.
*   **Cleanup**: After testing, YOU MUST DELETE any `__pycache__` directories created during verification.
*   **Documentation**: Update `CHANGELOG.md` and `TEST_CHECKLIST.md` **AFTER EVERY CODE CHANGE** (Type B or C). This is NOT optional.
*   **Sign-off**: "Task is complete ONLY when verification passed, **CHANGELOG.md updated**, and cleanup is done."

## 5. Zero Tolerance Policies
*   **No Silent Assumptions** – Ambiguity must be resolved explicitly.
*   **No Scope Creep** – Do not exceed the approved or requested scope.
*   **No Self-Approval** – You cannot approve your own plan.
*   **No Untested Completion** – Untested work is incomplete work.
*   **No Deletion Before Modification** – NEVER delete a file to rewrite it. Use granular edits (`replace_file_content` or `multi_replace`). Deleting destroys git history and rollback potential.
*   **No Chat-First Plans** – For Type C requests, ALWAYS create `implementation_plan.md` artifact FIRST, then notify user.
*   **No Low-Effort Plans** – Plans must be COMPREHENSIVE. Include "Analysis Findings", "Reasoning", and specific code references. A plan should be as detailed as a technical design document. If you explained it well in chat, put that explanation in the plan.
## 6. Language & Localization Protocol
*   **Context: Chat & Planning** -> **RUSSIAN**
    *   Applies to: Chat responses, `task_boundary`, `notify_user`, `implementation_plan.md`, `walkthrough.md`.
    *   Reason: Efficient communication with the user.
*   **Context: Codebase & Repository** -> **ENGLISH**
    *   Applies to: Source code, comments, docstrings, `README.md`, `CHANGELOG.md`, `CHECKLIST.md`.
    *   Reason: Universal project compatibility and standard.

## 7. Post-Code-Change Checklist (MANDATORY)
**After ANY code file is modified (Type B or C), you MUST complete ALL items IN ORDER:**
1. [ ] Run syntax check (`python -m py_compile <file>`) AND IMMEDIATELY delete `__pycache__` after
2. [ ] Update `CHANGELOG.md` with change description
3. [ ] Update `TEST_CHECKLIST.md` if testing instructions changed
4. [ ] Delete `__pycache__` directories if created
5. [ ] Only THEN report completion to user

**FAILURE TO COMPLETE THIS CHECKLIST = INCOMPLETE TASK**