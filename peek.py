# peek.py - Step 1: download the Twin-2K-500 dataset and look at one real person.

from datasets import load_dataset   # Hugging Face's downloader

# Download the "full persona" part of the dataset (about 200 MB).
# The first run downloads it; after that it loads from your Mac in seconds.
ds = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")
people = ds["data"]                 # the table of people

# How many real people are in the dataset?
print("Number of people:", len(people))

# Which pieces of information does each person have?
print("Columns:", people.column_names)
print()

# Show the start of the first person's description.
print("=== Person #1 (summary, first 2000 characters) ===")
print(people[0]["persona_summary"][:2000])
