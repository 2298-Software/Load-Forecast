import pandas as pd

df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }, dtype=str
)

df2 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A6", "A7"],
        "B": ["B0", "B5", "B6", "B7"],
        "C": ["C4", "C5", "C6", "C7"],
        "D": ["D4", "D5", "D6", "D7"],
    }, dtype=str
)

df1['ds'] = df1['A']
df2['ds'] = df2['A']

print(f'columns are {df1.columns}')
result=pd.concat([df1.set_index('ds'),df2.set_index('ds')], axis=1, join='outer')
print(result.to_string())