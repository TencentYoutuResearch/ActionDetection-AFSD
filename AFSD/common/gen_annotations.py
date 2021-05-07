import pandas as pd

data = pd.read_csv('thumos_annotations/val_Annotation.csv')
df = pd.DataFrame(data)

new_values = []
for d in df.values[:]:
    if d[2] != 0:
        new_values.append(d)

df2 = pd.DataFrame(new_values, columns=df.columns)
df2.to_csv('thumos_annotations/val_Annotation_ours.csv', index=False)

data = pd.read_csv('thumos_annotations/test_Annotation.csv')
df = pd.DataFrame(data)

new_values = []
for d in df.values[:]:
    if d[2] != 0:
        new_values.append(d)

df2 = pd.DataFrame(new_values, columns=df.columns)
df2.to_csv('thumos_annotations/test_Annotation_ours.csv', index=False)
