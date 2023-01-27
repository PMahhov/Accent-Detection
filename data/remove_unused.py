import pandas as pd

df = pd.read_csv('audio.csv', delimiter = ';')

df = df[df.primary_lang_group_id != 0]

df.to_csv('audio10.csv')