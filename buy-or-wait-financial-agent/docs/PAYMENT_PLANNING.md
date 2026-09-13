# Payment Planning

The planner considers the supported payment mechanisms available to each request.

Candidate classes include:

- full payment now
- partial payment when explicitly permitted
- supplied installment options
- full payment after eligible spending changes
- waiting until a safe future date
- not recommended when no safe completion path exists

A candidate is first validated for safety and deadline completion. Feasible candidates are then ranked deterministically. The ranking prioritizes completing by the deadline, avoiding spending changes, minimizing total paid, starting earlier, using fewer payments, and finally using the lowest payment-option identifier as the deterministic tie-breaker.
