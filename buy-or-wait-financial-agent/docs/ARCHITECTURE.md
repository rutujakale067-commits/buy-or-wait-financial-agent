# Architecture

## Design goal

Separate unstructured-data extraction from the financial decision engine so that financial safety checks remain deterministic and testable.

## Pipeline

1. Load request, profile, event, exchange-rate, payment-option, message, and image metadata.
2. Resolve missing event amounts from linked images using OCR where available.
3. Normalize monetary values into the user's home currency using supplied dated exchange rates.
4. Interpret supported message patterns into structured facts.
5. Detect conservative recurring transaction patterns from historical settled events.
6. Forecast daily cash flow for 90 days.
7. Generate payment candidates.
8. Simulate each candidate against the minimum safe balance and deadline.
9. Rank feasible candidates deterministically.
10. Emit the required output schema.

## Safety boundary

The financial engine does not assume missing money is zero and does not treat uncertain future credits as guaranteed cash. Candidate plans must pass the same simulator used for affordability checks.
