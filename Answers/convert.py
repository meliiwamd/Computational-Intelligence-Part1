import json

with open('CI992_HW1.ipynb', 'r') as f:
  notebook = json.load(f)
  
questions = ['Q1', 'Q2', 'Q4', 'Q5']
cells = notebook['cells']

for q in questions:
  with open(f"{q}.py", 'w') as f:
    f.write('')
  for cell in cells:
    if cell['cell_type'] == 'code' and f"{q}_graded" in cell['source'][0]:
      with open(f"{q}.py", 'a+') as f:
        f.writelines(cell['source'])
        f.write('\n\n')