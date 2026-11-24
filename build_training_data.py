# build_training_data.py - Step 4a: build the practice questions for fine-tuning (option B).
#
# For every TRAIN person: a SHORT description (decided in Step 4: demographics that matter for
# buying + 11 money/decision scores) + one product question + their REAL answer (Yes/No).
# Test people are never used here.

import glob, json, re, random
import pandas as pd

# ---- The decisions (edit these lists to run a different experiment) -------------
DEMOGRAPHICS = ["Geographic region", "Gender", "Age", "Education level", "Employment status",
                "Marital status", "Income", "Household size"]
# (left out on purpose: Race, Religion, Religious attendance, Political affiliation,
#  Political views, Citizen of the US)
SCORES = {                       # score name in the data -> plain-English label
    "score_ST-TW": "Spending style (tightwad 0 <-> spendthrift 100)",
    "score_finliteracy": "Financial literacy",
    "score_numeracy": "Numeracy",
    "score_riskaversion": "Risk aversion",
    "score_lossaversion": "Loss aversion",
    "score_discount": "Impatience",
    "score_presentbias": "Present bias",
    "score_mentalaccounting": "Mental accounting",
    "score_minimalism": "Minimalism",
    "score_maximization": "Maximizer (looks for the best option)",
    "wave1_score_conscientiousness": "Conscientiousness",
}
SYSTEM = ("You simulate one specific real person. Answer exactly as this person would, "
          "based on the description. Reply with Yes or No only.")

def short_description(summary):
    lines = []
    for field in DEMOGRAPHICS:
        m = re.search(rf"^{re.escape(field)}: (.*)$", summary, re.M)
        if m:
            value = m.group(1)
            if field == "Geographic region":
                value = value.split(" (")[0]          # "South (TX, OK, ...)" -> "South"
            lines.append(f"{field}: {value}")
    lines.append("Percentile among US adults (0 = lowest, 100 = highest):")
    for key, label in SCORES.items():
        m = re.search(rf"^{re.escape(key)} = \S+ \((\d+)\w\w percentile\)", summary, re.M)
        if m:
            lines.append(f"- {label}: {m.group(1)}")
    return "\n".join(lines)

def question_prompt(description, question):
    return (description + "\n\n### QUESTION\n" + question +
            "\nAnswer as this person: Yes (I would purchase) or No (I would not purchase)?")

if __name__ == "__main__":
    # ---- 1. Short description for EVERY person (the test script needs them too) ----
    summaries = pd.concat(pd.read_parquet(p, columns=["pid", "persona_summary"])
                          for p in sorted(glob.glob("data/full_persona/chunks/*.parquet")))
    summaries["pid"] = summaries.pid.astype(str)
    summaries["short"] = summaries.persona_summary.map(short_description)
    summaries[["pid", "short"]].to_csv("data/short_descriptions.csv", index=False)
    short_of = dict(zip(summaries.pid, summaries.short))

    # ---- 2. Train people only; set 100 of them aside as "validation" -------------
    # Validation = practice-exam people: the training tool checks on them while training
    # to see whether the model learns general patterns or just memorises its practice people.
    table = pd.read_csv("data/pricing_table_split.csv")
    table["pid"] = table.pid.astype(str)
    train = table[table.group == "train"]
    train_people = sorted(train.pid.unique(), key=int)
    random.Random(42).shuffle(train_people)
    valid_people = set(train_people[:100])

    # ---- 3. One example per (person, product) -------------------------------------
    examples = {"train": [], "valid": []}
    for _, row in train.iterrows():
        ex = {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question_prompt(short_of[row.pid], row.question)},
            {"role": "assistant", "content": "Yes" if row.buy_second_time == 1 else "No"},
        ]}
        examples["valid" if row.pid in valid_people else "train"].append(ex)

    import os
    os.makedirs("data/training", exist_ok=True)
    for name, exs in examples.items():
        random.Random(42).shuffle(exs)              # mix people and products
        with open(f"data/training/{name}.jsonl", "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")

    print(f"train: {len(examples['train'])} examples from {len(train_people) - 100} people")
    print(f"valid: {len(examples['valid'])} examples from 100 people")
    print("Yes share in train:", round(sum(e['messages'][2]['content'] == 'Yes'
                                           for e in examples['train']) / len(examples['train']) * 100, 1), "%")
    print("\n=== One example ===")
    for m in examples["train"][0]["messages"]:
        print(f"[{m['role'].upper()}]\n{m['content']}\n")
