import json

with open('notebooks/exploration.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of the cell that has "## 4. Evaluation Metrics"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and '## 4. Evaluation Metrics' in cell['source'][0]:
        insert_idx = i + 1
        break

if insert_idx != -1:
    new_cell = {
      "cell_type": "markdown",
      "id": "new_ground_truth_note",
      "metadata": {},
      "source": [
       "**Note on Ground Truth:** In this evaluation, a \"true match\" is defined strictly using an independent, rule-based approach. A student is considered a true match if they meet *all* of the job's minimum requirements: their average score for the required skills must meet or exceed the job's average minimum skill requirement, their CGPA must be at least the job's minimum CGPA, and their total experience units must be greater than or equal to the job's experience requirement."
      ]
    }
    # Check if already inserted
    already_inserted = False
    for c in nb['cells']:
        if c['cell_type'] == 'markdown' and len(c['source']) > 0 and 'Note on Ground Truth' in c['source'][0]:
            already_inserted = True
            break
            
    if not already_inserted:
        nb['cells'].insert(insert_idx, new_cell)
        print("Inserted cell.")
    else:
        print("Cell already exists.")

with open('notebooks/exploration.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
