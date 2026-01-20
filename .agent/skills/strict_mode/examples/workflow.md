# Example: Strict Protocol Workflow

This example demonstrates the correct interaction flow when using the `strict-protocol` skill.

## User Request
> "Refactor the authentication module to use OAuth2."

## 1. Stop & Plan
The agent analyzes the request and creates `implementation_plan.md`.

```markdown
# Implementation Plan - OAuth2 Migration
## Goal
Replace Basic Auth with OAuth2.
## Changes
- [MODIFY] auth.py
...
```

## 2. Ask for Approval
The agent uses `notify_user` to present the plan.

> **Agent**: "I have created a plan to migrate to OAuth2. It involves editing `auth.py`. Do you approve?"

## 3. Wait
The agent does NOTHING until the user responds.

> **User**: "Yes, proceed."

## 4. Execute
The agent switches to EXECUTION mode and applies the changes defined in the plan.
