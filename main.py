import pandas as pd

file_path = "data/marketing_campaign.csv"

df = pd.read_csv(file_path, sep="\t")

print("Pierwsze 5 wierszy:")
print(df.head())

print("\nInformacje o danych:")
print(df.info())

print("\nNazwy kolumn:")
print(df.columns.tolist())

print("\nLiczba braków w każdej kolumnie:")
print(df.isnull().sum())

print("\nStatystyki opisowe:")
print(df.describe(include="all"))