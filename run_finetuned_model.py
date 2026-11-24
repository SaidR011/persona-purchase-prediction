# run_finetuned_model.py - Step 5: score a model (optionally + a trained LoRA adapter)
# on the TEST people, using the SHORT descriptions (same format as training).
#
# Usage:
#   python run_finetuned_model.py --adapter adapters/test_run --people 50
#   python run_finetuned_model.py --adapter adapters/main_run              (all 500)
#   python run_finetuned_model.py --adapter adapters/main_run --stranger   (control)
#   python run_finetuned_model.py --people 50                              (no adapter = untrained)

import argparse, time, os
import pandas as pd
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from build_training_data import SYSTEM, question_prompt   # EXACT same text as training

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default=None, help="folder with a trained LoRA adapter")
parser.add_argument("--people", type=int, default=500)
parser.add_argument("--stranger", action="store_true", help="control: use another person's description")
args = parser.parse_args()

table = pd.read_csv("data/pricing_table_split.csv")
test = table[table.group == "test"].copy()
test["pid"] = test.pid.astype(str)
short = pd.read_csv("data/short_descriptions.csv")
short_of = dict(zip(short.pid.astype(str), short.short))

people = sorted(test.pid.unique(), key=int)[: args.people]
description_for = {p: short_of[people[(i + 1) % len(people)] if args.stranger else p]
                   for i, p in enumerate(people)}

print("Loading model" + (f" + adapter {args.adapter}" if args.adapter else " (no adapter)") + "...")
model, tokenizer = load(MODEL, adapter_path=args.adapter)
mx.set_cache_limit(2 * 1024**3)          # memory fix from Step 3
YES = tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = tokenizer.encode("No", add_special_tokens=False)[0]

def text_before_answer(description, question):
    # Build the conversation EXACTLY like a training example (answer = "Yes"),
    # then cut it just before the answer. Training and testing see identical text.
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question_prompt(description, question)},
                {"role": "assistant", "content": "Yes"}]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    return full[: full.rfind("Yes")]

rows, start = [], time.time()
for n, pid in enumerate(people, 1):
    qs = test[test.pid == pid]
    prefix_text = text_before_answer(description_for[pid], "@@SPLIT@@").split("@@SPLIT@@")[0]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    cache = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=cache)                    # read the description once
    for _, q in qs.iterrows():
        rest_ids = tokenizer.encode(text_before_answer(description_for[pid], q.question)[len(prefix_text):],
                                    add_special_tokens=False)
        logits = model(mx.array([rest_ids]), cache=cache)[0, -1]
        p_yes = mx.softmax(mx.array([logits[YES], logits[NO]]))[0].item()
        trim_prompt_cache(cache, len(rest_ids))
        rows.append({"pid": pid, "qid": q.qid, "product": q["product"], "price": q.price,
                     "p_yes": round(p_yes, 4), "pred": int(p_yes > 0.5),
                     "truth": q.buy_second_time, "first_time": q.buy_first_time})
    del cache
    mx.clear_cache()                                              # memory fix from Step 3
    if n % 10 == 0 or n == len(people):
        d = pd.DataFrame(rows)
        print(f"{n}/{len(people)} people | accuracy so far {(d.pred == d.truth).mean()*100:.2f}% | {time.time()-start:.0f}s")

out = pd.DataFrame(rows)
tag = (os.path.basename(args.adapter.rstrip("/")) if args.adapter else "untrained_short") + ("_stranger" if args.stranger else "")
os.makedirs("results", exist_ok=True)
out.to_csv(f"results/{tag}_{len(people)}people.csv", index=False)

acc = (out.pred == out.truth).mean() * 100
ceiling = (out.first_time == out.truth).mean() * 100
print(f"\nRESULT ({tag}, {len(people)} people, {len(out)} answers)")
print(f"  Model accuracy:              {acc:.2f}%")
print(f"  Human ceiling (same people): {ceiling:.2f}%")
print(f"  Model as % of ceiling:       {acc / ceiling * 100:.1f}%")
print(f"  Bar to beat (price rule, all 500): 73.07%")
print(f"  Model says Yes {out.pred.mean()*100:.1f}% of the time (real people: {out.truth.mean()*100:.1f}%)")
out["price band"] = pd.cut(out.price, [-0.01, 2, 5, 10, 20, 100], labels=["$0-2", "$2-5", "$5-10", "$10-20", "$20+"])
g = out.groupby("price band", observed=True)[["truth", "pred"]].mean().mul(100).round(1)
g.columns = ["real people buy %", "model says yes %"]
print("\n" + g.to_string())
