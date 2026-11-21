# run_base_model.py - Step 3: ask an UNTRAINED small model to predict each test person's
# Yes/No purchase decisions, and score it against the real answers.
#
# Usage:
#   python run_base_model.py --people 10      (quick try on 10 test people)
#   python run_base_model.py                  (all 500 test people)

import argparse, glob, time
import pandas as pd
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

parser = argparse.ArgumentParser()
parser.add_argument("--people", type=int, default=500, help="how many test people to run")
parser.add_argument("--stranger", action="store_true",
                    help="control: give each person ANOTHER person's description")
args = parser.parse_args()

# ---- 1. Load the test questions and the person descriptions ----------------
table = pd.read_csv("data/pricing_table_split.csv")
test = table[table.group == "test"].copy()
test["pid"] = test.pid.astype(str)

summaries = pd.concat(pd.read_parquet(p, columns=["pid", "persona_summary"])
                      for p in sorted(glob.glob("data/full_persona/chunks/*.parquet")))
summaries["pid"] = summaries.pid.astype(str)
persona_of = dict(zip(summaries.pid, summaries.persona_summary))

people = sorted(test.pid.unique(), key=int)[: args.people]
# Stranger control: person i gets the description of the next person in the list.
description_for = {p: persona_of[people[(i + 1) % len(people)] if args.stranger else p]
                   for i, p in enumerate(people)}

# ---- 2. Load the model ------------------------------------------------------
print("Loading model (first time downloads about 2.3 GB)...")
model, tokenizer = load(MODEL)
# Memory fix: Apple's MLX keeps used memory in a reserve "for later" instead of giving it back.
# Cap that reserve at 2 GB so the Mac doesn't run out of memory.
mx.set_cache_limit(2 * 1024**3)
YES = tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = tokenizer.encode("No", add_special_tokens=False)[0]

def build_prompt(description, question):
    messages = [
        {"role": "system", "content": "You simulate one specific real person. "
         "Answer exactly as this person would, based on the description. Reply with Yes or No only."},
        {"role": "user", "content": description + "\n\n### QUESTION\n" + question +
         "\nAnswer as this person: Yes (I would purchase) or No (I would not purchase)?"},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ---- 3. Predict -------------------------------------------------------------
# Speed trick: the long description is the same for all 40 questions of a person,
# so the model reads it ONCE (kept in "cache"), then reads only each short question.
rows, start = [], time.time()
for n, pid in enumerate(people, 1):
    qs = test[test.pid == pid]
    full = build_prompt(description_for[pid], "@@SPLIT@@")
    prefix_text = full.split("@@SPLIT@@")[0]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    cache = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=cache)               # read the description once

    for _, q in qs.iterrows():
        rest_text = build_prompt(description_for[pid], q.question)[len(prefix_text):]
        rest_ids = tokenizer.encode(rest_text, add_special_tokens=False)
        logits = model(mx.array([rest_ids]), cache=cache)[0, -1]
        p_yes = mx.softmax(mx.array([logits[YES], logits[NO]]))[0].item()
        trim_prompt_cache(cache, len(rest_ids))              # forget this question
        rows.append({"pid": pid, "qid": q.qid, "product": q["product"], "price": q.price,
                     "p_yes": round(p_yes, 4), "pred": int(p_yes > 0.5),
                     "truth": q.buy_second_time, "first_time": q.buy_first_time})

    # Memory fix: throw away this person's cache and hand the memory back before the next person.
    del cache
    mx.clear_cache()

    if n % 10 == 0 or n == len(people):
        done = pd.DataFrame(rows)
        acc = (done.pred == done.truth).mean() * 100
        print(f"{n}/{len(people)} people | accuracy so far {acc:.2f}% | {time.time()-start:.0f}s")

# ---- 4. Score and save ------------------------------------------------------
out = pd.DataFrame(rows)
name = "base_model_stranger" if args.stranger else "base_model"
out.to_csv(f"results/{name}_{len(people)}people.csv", index=False)

acc = (out.pred == out.truth).mean() * 100
ceiling = (out.first_time == out.truth).mean() * 100
print()
print(f"RESULT ({name}, {len(people)} people, {len(out)} answers)")
print(f"  Model accuracy:          {acc:.2f}%")
print(f"  Human ceiling (same people): {ceiling:.2f}%")
print(f"  Model as % of ceiling:   {acc / ceiling * 100:.1f}%")
print(f"  Bar to beat (price rule, all 500): 73.07%")
print(f"  Model says Yes {out.pred.mean()*100:.1f}% of the time (real people: {out.truth.mean()*100:.1f}%)")
