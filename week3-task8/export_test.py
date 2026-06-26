import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')
train_val, test = train_test_split(df, test_size=0.15, random_state=42)

test.to_csv('d:/Placemux-aiml/week3-task8/data/test_split.csv', index=False)
print(f"Test data saved! Rows: {len(test)}")
