from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
from io import StringIO 

app = FastAPI()

class DataPreprocessor:
    def __init__(self, data):
        if isinstance(data, dict): 
            self.df = pd.DataFrame(data, index=[0])
        else:
            self.df = pd.DataFrame(data)

    def clean_data(self):
        self.df['mileage'] = self.df['mileage'].str[:-5].str.strip()
        self.df['engine'] = self.df['engine'].str[:-3].str.strip()
        self.df['max_power'] = self.df['max_power'].str[:-4].str.strip()
        self.df['max_power'] = self.df['max_power'].replace('', '0')
        self.df[['mileage', 'engine', 'max_power', 'seats']] = self.df[['mileage', 'engine', 'max_power', 'seats']].astype(float)

      
        medians =  pickle.load(open('median_values.pkl', 'rb'))
        # Заполняем пропуски средними значениями
        for col in ['mileage', 'engine', 'max_power', 'seats']:
            self.df[col] = self.df[col].fillna(medians[col])

        self.df[['mileage', 'engine', 'max_power', 'seats']] = self.df[['mileage', 'engine', 'max_power', 'seats']].astype(float)
        self.df[['engine', 'seats']] = self.df[['engine', 'seats']].astype(int)

        return self

    def drop_torque(self):
        # Удаляем столбец torque
        self.df = self.df.drop(columns='torque')

        return self

    def encode(self):
        # кодирование категориальных переменных
        mean_selling_price = pickle.load(open('mean_selling_price.pkl', 'rb'))
        # Применяем среднее значение к столбцу name
        self.df['name_encoded'] = self.df['name'].map(mean_selling_price)

        # Заменим значения отсутствующие в mean_selling_price на среднее значение по mean_selling_price
        mean_selling_price = mean_selling_price.reset_index()
        self.df['name_encoded']  = self.df['name_encoded'].fillna(mean_selling_price['selling_price'].mean())

        # загрузка OneHotEncoder
        encoder = pickle.load(open('encoder.pkl', 'rb'))

        encoded_data = encoder.transform(self.df[['seats', 'fuel', 'seller_type', 'transmission', 'owner']])
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['seats', 'fuel', 'seller_type', 'transmission', 'owner']))

        self.df = pd.concat([self.df.drop(columns=['seats', 'fuel', 'seller_type', 'transmission', 'owner']),
                            encoded_df], axis=1)

        # Отбор финальных колонок
        self.df = self.df.select_dtypes(include=['int', 'float', 'bool']).copy()
        self.df_features = self.df.drop('selling_price', axis=1)
   
        return self.df_features

    def get_target(self):
        return self.df['selling_price']



class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

model = pickle.load(open('model.pkl', 'rb'))

@app.post("/predict_item")
def predict_item(item: Item) -> str:
    # Подготовка данных
    data = item.dict()
    preprocessor = DataPreprocessor(data)
    preprocessed_data = preprocessor.clean_data().drop_torque().encode()

    true_target = preprocessor.get_target()

    # Предсказание
    prediction = model.predict(preprocessed_data)

    return f'prediction price: {float(prediction[0]):.2f} real price: {true_target.iloc[0]:.2f}'


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    # Чтение CSV файла
    content = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(content))

    # Обработка данных
    preprocessor = DataPreprocessor(df)
    preprocessed_data = preprocessor.clean_data().drop_torque().encode()
    
    # Предсказание для всех объектов
    predictions = model.predict(preprocessed_data)

    # Добавляем предсказания в DataFrame
    df['predictions'] = predictions

    # Сохранение результата в новый файл
    output_file = "d:\Рабочий стол\ml/predictions_output.csv" 
    df.to_csv(output_file, index=False)

    return {"filename": output_file}
