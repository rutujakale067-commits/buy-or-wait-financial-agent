# Financial Engine

The engine represents cash flow as a daily timeline. Starting balance is combined with normalized future cash events and conservative recurring projections.

For a candidate plan, scheduled payments are applied on their dates and the resulting balance is checked across the forecast horizon. A plan is feasible only when the balance remains at or above the configured minimum safe balance and the required payment is completed by its deadline.

Recurring groups are inferred from repeated historical settled transactions. The implementation uses minimum occurrence and interval-consistency thresholds so one-off transactions are not projected as recurring spending.

Invalid transaction statuses and non-cash/unrealized items are excluded from the safety forecast according to the challenge data model.
