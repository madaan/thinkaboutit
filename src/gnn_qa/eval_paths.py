import pandas as pd
import sys
data = pd.read_csv(sys.argv[1], sep="\t")
acc = 0.
sim = 0.

def paths_overlap(p1, p2):
  p1 = set(p1.split("->"))
  p2 = set(p2.split("->"))
  return len(p1.intersection(p2)) * 100 / len(p1.union(p2))

for i, row in data.iterrows():
  true_paths = set([p.strip() for p in row["true_paths"].split("||")])
  actual_path = row["actual_paths"]
  print(true_paths, row["actual_paths"])
  if row["actual_paths"] in true_paths:
    acc += 1
  sim += sum([paths_overlap(true_path, actual_path) for true_path in true_paths]) / len(true_paths)

print(round(acc * 100 / len(data), 2))
print(round(sim / len(data), 2))