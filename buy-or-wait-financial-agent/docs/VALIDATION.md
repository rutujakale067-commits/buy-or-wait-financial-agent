# Validation

Validation was performed at multiple levels:

1. **Output schema validation** — required columns, row count, and value bounds.
2. **Deterministic simulation** — candidate plans are rechecked against the same cash-flow safety model used by the decision engine.
3. **Constraint checks** — payment schedules, deadlines, payment-method eligibility, and safe-payment bounds are validated.
4. **Sample-data verification** — the supplied sample outputs were used as a development reference.

The solution does not claim an independent ground-truth score for every competition request; final challenge scoring remains the authoritative evaluation.
