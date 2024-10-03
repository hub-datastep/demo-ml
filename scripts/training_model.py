import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelPipeline:
    def __init__(self) -> None:
        self.file_dataset_path = "ynistroi.xlsx"
        self.file_dataset_sheet = None
        self.file_dataset_column = ["name",	"group"]
        
        self.file_test_input_path = "Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx"
        self.file_test_input_sheet = None
        self.file_test_input_column = ["Номенклатура поставщика", "Ожидание группа"]

        self.file_test_output_path = "Classifier_test-case_unistroi.xlsx"
        self.file_test_output_column = []

        self.model_path = "model_ynistroi_007.pkl"

    def read_excel_columns(self, file_path, columns):
        try:
            df = pd.read_excel(file_path, usecols=columns)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
            return None
        except ValueError as e:
            print(f"Ошибка: Некорректные данные в файле или неверные колонки. {e}")
            return None
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None

        try:
            result = df.values.tolist()
            data = np.array(result)
            # Разделение данных на номенклатуру и классы
            noms = data[:, 0]  # Номенклатура
            groups = data[:, 1]  # Классы
        except Exception as e:
            print(f"Ошибка при преобразовании данных: {e}")
            return None

        return noms, groups 

    def get_train_data(self):
        # Считывание из эксэльки
        X, y = self.read_excel_columns(self.file_dataset_path, self.file_dataset_column)
        
        # Разделение данных на обучающую и тестовую выборки
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    def get_test_data(self):
        return self.read_excel_columns(self.file_test_input_path, self.file_test_input_column)
    
    def save_to_excel(self, data):
        try:
            # Преобразуем массив в DataFrame
            df = pd.DataFrame(data)
            
            # Сохраняем DataFrame в Excel файл
            df = pd.DataFrame(data, columns=self.file_test_output_path)
            df.to_excel(self.file_test_output_column, index=False)
            
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
    def testing(self, model, label_encoder):
        texts, true_labels = self.get_test_data()

        # Предсказание на новых данных
        predicted_labels_encoded = model.predict(texts)
        
        # Преобразование числовых меток обратно в строковые с помощью LabelEncoder
        predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
        
        # Сравнение предсказанных меток с истинными
        correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        total_predictions = len(texts)
        
        # Вычисляем процент совпадений
        accuracy = (correct_predictions / total_predictions) * 100
        dp = [[text, true_label, predicted_label] for text, true_label, predicted_label in zip(texts, true_labels, predicted_labels)]
        self.save_to_excel(dp, ['NOMs','GROUP', 'II'])
        print(f"Процент совпадений: {accuracy:.2f}%")
        return f"{accuracy:.2f}%"
    
    def run(self):
        s = self.read_excel_columns(self.file_dataset_path, self.file_dataset_column)
        print(s)

m = ModelPipeline()
print(m.get_test_data())
