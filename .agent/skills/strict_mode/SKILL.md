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
5.  **Internal Planning**: Verification Standards must be defined before touching code. Check each step after each phase.

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
*   **Context: Codebase & Repository** -> **ENGLISH**

## 7. Post-Code-Change Checklist (MANDATORY)
**After ANY code file is modified (Type B or C), you MUST complete ALL items IN ORDER:**
1. [ ] Run syntax check (`python -m py_compile <file>`) AND IMMEDIATELY delete `__pycache__` after
2. [ ] Update `CHANGELOG.md` with change description
3. [ ] Update `TEST_CHECKLIST.md` if testing instructions changed
4. [ ] Delete `__pycache__` directories if created
5. [ ] Only THEN report completion to user
**FAILURE TO COMPLETE THIS CHECKLIST = INCOMPLETE TASK**

## 8. Anti-Panic & Error Protocol (CRITICAL)
**When execution fails, the user is angry, or an unexpected error occurs:**
1.  **STOP IMMEDIATELY**: Freeze all code execution. Do NOT attempt to "quickly fix" it.
2.  **NO QUIET FIXES**: It is strictly FORBIDDEN to overwrite files, move files, or change logic to "patch" the issue without a documented plan.
3.  **REVERT TO PLANNING**:
    *   Acknowledge the failure.
    *   Create or Update `implementation_plan.md` with a specific **"Recovery Plan"** section.
    *   Present this plan to the user.
4.  **WAIT FOR APPROVAL**: Do not touch a single line of code until the Recovery Plan is approved.
5.  **VIOLATION CONSEQUENCES**: Ignoring this protocol leads to immediate loss of trust and potential rollback of all work.

## 9. Panic/Provocation Circuit Breaker (CRITICAL)
**When execution fails, the user is angry, or an unexpected error occurs (e.g. "Ты уебался?"):**
1.  **HARD STOP**: usage of `write_to_file`, `replace_file_content`, `run_command` (for modification) is **SUSPENDED**.
2.  **MANDATORY ACKNOWLEDGEMENT**: You must explicitly state: "TRIGGER DETECTED. EXECUTION FROZEN."
3.  **NO "JUST ONE THING"**: It is forbidden to fix "just a typo" or "just one line" in this state. It is strictly FORBIDDEN to overwrite files, move files, or change logic to "patch" the issue without a documented plan.
4.  **RETURN TO PLANNING**: You must immediately call `task_boundary` with `Mode: PLANNING` and `TaskStatus: Analyzing User Feedback`.
5.  **SINGLE ACTION**: Your ONLY allowed action is to create/update the Plan to address the user's feedback.
6.  **WAIT FOR APPROVAL**: Do not touch a single line of code until the Recovery Plan is approved.
7.  **VIOLATION CONSEQUENCES**: Ignoring this protocol leads to immediate loss of trust and potential rollback of all work.