# AI Usage

## Development-time AI assistance

ChatGPT was used during development for:

- architecture brainstorming
- identifying edge cases
- reasoning about financial constraints
- documentation drafting
- interview preparation

## Runtime boundary

The submitted runtime does not call an external LLM API. The runtime uses deterministic Python logic, regular-expression-based structured message parsing, optional OCR for linked images, transaction analysis, forecasting, simulation, and plan ranking.

## Production improvement

A useful next step would be a lightweight LLM or fine-tuned classifier for robust extraction of financial facts from varied natural-language messages. Extracted facts should then enter the existing deterministic financial engine, preserving auditable affordability decisions.
