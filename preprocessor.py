import pandas as pd
import numpy as np


'''
1) Заполняет пропуски (на ваш выбор как, я это оценю)
2) Удаляет высокоскорреллированные фичи
3) Считает среднее по каждой числовой фиче
4) Обрабатывает категориальные фичи. Важно: и не только те, которые представлены типом object, но и числовые, у которых менее 25 уникальных значений
5) Разбивает колонку со временем (create_dttm) на год, месяц и день
Данных заранее не даю. Сервис должен работать на любых данных, где есть колонка create_dttm
Отправить ссылку на репозиторий с кодом мне в лс
'''

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_features = []
        self.num_features = []
        self.num_means = {}

    def set_features(self):
        self.num_features = self.df.select_dtypes(include='number').columns.tolist()
        self.cat_features = self.df.select_dtypes(include='object').columns.tolist()

    # number 1
    def fill_missing(self):
        for col in self.num_features:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        for col in self.cat_features:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    # number 2
    def remove_corr(self, threshold=0.9):
        corr_matrix = self.df[self.num_features].corr().abs()
        np_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [col for col in np_triu.columns if any(np_triu[col] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
        self.num_features = [col for col in self.num_features if col in self.df.columns]

    # number 3
    def means(self):
        self.num_means = {}
        for col in self.num_features:
            self.num_means[col] = self.df[col].mean()

        return self.num_means

    # number 4
    def encode_categorial(self):
        cat_cols =  self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.num_features:
            if self.df[col].nunique() < 25:
                if col not in cat_cols:
                    cat_cols.append(col)

        self.cat_features = cat_cols

        self.df = pd.get_dummies(self.df, columns=self.cat_features, drop_first=True)

    #number 5
    def dates(self, col='create_dttm'):
        self.df[col] = pd.to_datetime(self.df[col], dayfirst=True)

        # new columns
        self.df['Year'] = self.df[col].dt.year
        self.df['Month'] = self.df[col].dt.month
        self.df['Day'] = self.df[col].dt.day

        self.df.drop(columns=[col], inplace=True)

    def __str__(self):
        return f'{self.df.head(5)}'
