# Pre-registration: Can a small fine-tuned model predict who buys at what price?

Written BEFORE any model is run. Once committed to GitHub, this file is not edited.
(Follows Talp's rule 1: "Write it down first".)

## Data
- Twin-2K-500 (Toubia et al., 2025; CC-BY-4.0): 2,058 real US adults.
- Task: 40 grocery products, each shown at a randomly assigned price.
  "Would you or would you not purchase this product?"  (Yes / No)
- Persona for each person: their waves 1-3 survey answers (demographics, personality,
  economic preferences, etc.). The 40 purchase answers are NOT in the persona.
- Truth: the person's real wave 4 answer.

## Split (fixed, random_state=42)
- 1,558 train people, 500 test people (20,000 test answers).
- The model never sees any test person during training.

## Scores already known (no AI involved)
| Predictor | Accuracy on test |
|---|---|
| Human ceiling (same person, same question, two weeks earlier) | 84.54% |
| Lazy guess A: always "No" | 56.29% |
| Lazy guess B: product + price rule from train people | 73.07% |

## What counts as success (decided before any model is run)
1. MAIN TEST: the fine-tuned model must beat Lazy guess B (73.07%) on the 500 test people.
   If it does not, the result is reported as a failure.
2. PERSONA TEST: the fine-tuned model must score higher with each person's real description
   than with a random stranger's description. If it does not, the persona is not helping.
3. For comparison, the untrained base model is also scored, with the same rules.

## Rules
- Every number is reported, including losses.
- Each model result is the average of 3 runs from scratch.
- Results are reported as raw accuracy AND as % of the human ceiling.
