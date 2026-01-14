# Example: Code Review Feedback

## Bad Review
"Fix this."
(Vague, no reasoning)

## Good Review
"This loop has O(N^2) complexity which might slow down character loading. Since `all_characters` is potentially large, consider pre-calculating a lookup map for O(1) access.

Example:
```python
char_map = {c.id: c for c in all_characters}
```
"
(Specific, reasoned, alternative suggested)
