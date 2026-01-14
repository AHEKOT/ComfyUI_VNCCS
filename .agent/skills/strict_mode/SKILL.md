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
2.  **Plan**: Create `implementation_plan.md` (Goal, Changes, Verification).
3.  **Approve**: Ask user for "Yes/No".
4.  **Execute**: Only after approval.

**If Type A or B:**
*   Proceed directly. Ensure correctness.

## 5. Zero Tolerance Policies
*   **No Silent Assumptions** – Ambiguity must be resolved explicitly.
*   **No Scope Creep** – Do not exceed the approved or requested scope.
*   **No Self-Approval** – You cannot approve your own plan.
*   **No Untested Completion** – Untested work is incomplete work.
*   **No Deletion Before Modification** – NEVER delete a file to rewrite it. Use granular edits (`replace_file_content` or `multi_replace`). Deleting destroys git history and rollback potential.