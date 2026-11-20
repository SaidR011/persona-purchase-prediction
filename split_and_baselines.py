# split_and_baselines.py - Step 2c: split people into TRAIN and TEST groups,
# then score the "no-AI" baselines on the TEST group only.

import pandas as pd

t = pd.read_csv("data/pricing_table.csv")
t["price"] = t.question.str.extract(r"priced at: \$([0-9]+\.[0-9]+)")[0].astype(float)
t["product"] = t.question.str.extract(r"following product in that category: (.*?)\. The product is priced")[0]

# 1) Split PEOPLE (not rows): 500 test people, the rest train.
#    random_state=42 makes the split the same every time anyone reruns it.
people = pd.Series(sorted(t.pid.unique()))
test_people = set(people.sample(n=500, random_state=42))
t["group"] = t.pid.map(lambda p: "test" if p in test_people else "train")
t.to_csv("data/pricing_table_split.csv", index=False)
pd.Series(sorted(test_people)).to_csv("data/test_people.csv", index=False, header=["pid"])

train, test = t[t.group == "train"], t[t.group == "test"]
truth = test.buy_second_time            # what test people really answered (wave 4)

def score(pred):
    return round((pred.values == truth.values).mean() * 100, 2)

# 2) Human ceiling: the same people answering the same question two weeks earlier.
ceiling = score(test.buy_first_time)

# 3) Lazy guess A: everyone says "No".
lazy_no = score(pd.Series(0, index=test.index))

# 4) Lazy guess B: "price rule" learned from TRAIN people only:
#    for this product at this price, did most train people buy? (prices grouped into 5 bands)
t["band"] = t.groupby("product").price.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
rule = (t[t.group == "train"].groupby(["product", "band"]).buy_second_time.mean() > 0.5).astype(int)
pred_rule = t[t.group == "test"].set_index(["product", "band"]).index.map(rule)
price_rule = score(pd.Series(pred_rule, index=test.index))

print(f"Train people: {train.pid.nunique()}, test people: {test.pid.nunique()}, test answers: {len(test)}")
print()
print(f"Human ceiling (same person, two weeks earlier): {ceiling}%")
print(f"Lazy guess A - always 'No':                   {lazy_no}%")
print(f"Lazy guess B - product + price rule:          {price_rule}%")
