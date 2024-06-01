import pandas as pd

dictionary = pd.read_pickle(r'C:\MaH_Kretschmer\data\train_data\DEMO_EMO\dictionary.pkl')
cinfo = (list(dictionary.values()), list(dictionary.keys()))
print(cinfo)