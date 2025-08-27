import pandas as pd

from preprocessor import Preprocessor


df = pd.read_csv('')

preprocessor = Preprocessor(df)
preprocessor.set_features()
preprocessor.fill_missing()
preprocessor.remove_corr()
means = preprocessor.means()
preprocessor.dates()
preprocessor.encode_categorial()

print(preprocessor)

