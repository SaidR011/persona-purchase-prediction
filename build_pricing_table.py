# build_pricing_table.py - Step 2b: pull the 40 "would you buy this product at this price?"
# questions out of the dataset into one simple table: one row = one person x one product.

import glob, json
import pandas as pd

rows = []
for path in sorted(glob.glob("data/wave_split/chunks/*.parquet")):
    df = pd.read_parquet(path)
    for _, person in df.iterrows():
        # Answers to the SAME questions: first time (waves 1-3) and second time (wave 4).
        first = json.loads(person["wave4_Q_wave1_3_A"])
        second = json.loads(person["wave4_Q_wave4_A"])

        def pricing_questions(blocks):
            out = {}
            for block in blocks:
                if "Pricing" in block["BlockName"]:
                    for q in block["Questions"]:
                        if q["QuestionType"] == "MC":
                            out[q["QuestionID"]] = q
            return out

        q1, q2 = pricing_questions(first), pricing_questions(second)
        for qid, q in q2.items():
            if qid not in q1:
                continue
            rows.append({
                "pid": person["pid"],
                "qid": qid,
                "question": q["QuestionText"],
                "same_text_both_times": q["QuestionText"] == q1[qid]["QuestionText"],
                # 1 = "Yes, I would purchase", 0 = "No"
                "buy_first_time": int(q1[qid]["Answers"]["SelectedByPosition"] == 1),
                "buy_second_time": int(q["Answers"]["SelectedByPosition"] == 1),
            })

table = pd.DataFrame(rows)
table.to_csv("data/pricing_table.csv", index=False)
print("Saved data/pricing_table.csv with", len(table), "rows,",
      table.pid.nunique(), "people,", table.qid.nunique(), "products")
print("Question text identical both times:", table.same_text_both_times.mean())
