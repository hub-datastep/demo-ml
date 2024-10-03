import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

class ModelPipeline:
    def __init__(self) -> None:
        self._file_dataset_path = f"datasets/ynistroi.xlsx"
        self._file_dataset_sheet = "Лист1"
        self._file_dataset_column = ["name", "group"]
        
        self._file_test_input_path = f"test-sets/Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx"
        self._file_test_input_sheet = "test-cases"
        self._file_test_input_column = ["Номенклатура поставщика", "Ожидание группа"]

        self._file_test_output_path = f"output_test/Classifier_test-case_unistroi.xlsx"
        self._file_test_output_column = ['NOMs','GROUP', 'II', 'T|F']

        self._model_path = "model/model_unistroi.pkl"

        self._model = 0
        self._label_encoder = 0
        self._report=0
        self._conf_matrix=0
        self._execution_time=0
        self._accuracy=0

        self._X_train = 0
        self._X_test = 0
        self._y_train = 0
        self._y_test = 0
        self._texts = 0
        self._true_labels = 0

    def read_excel_columns(self, file_path, columns, sheet):
        try:
            df = pd.read_excel(file_path, usecols=columns, sheet_name=sheet)
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
        X, y = self.read_excel_columns(self._file_dataset_path, self._file_dataset_column, self._file_dataset_sheet)
        
        # Разделение данных на обучающую и тестовую выборки
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Encode labels
        le = LabelEncoder()
        self._y_train = le.fit_transform(self._y_train)
        self._y_test = le.transform(self._y_test)
        self._label_encoder = le

    def get_test_data(self):
        self._texts, self._true_labels = self.read_excel_columns(self._file_test_input_path, self._file_test_input_column, self._file_test_input_sheet)
    
    def save_to_excel(self, data):
        try:
            # Преобразуем массив в DataFrame
            df = pd.DataFrame(data)
            
            # Сохраняем DataFrame в Excel файл
            df = pd.DataFrame(data, columns=self._file_test_output_column)
            df.to_excel(self._file_test_output_path, index=False)
            
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    
    def testing(self):
        texts, true_labels = self._texts, self._true_labels

        # Предсказание на новых данных
        predicted_labels_encoded = self._model.predict(texts)
        
        # Преобразование числовых меток обратно в строковые с помощью LabelEncoder
        predicted_labels = self._label_encoder.inverse_transform(predicted_labels_encoded)
        
        # Сравнение предсказанных меток с истинными
        correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        total_predictions = len(texts)
        
        # Вычисляем процент совпадений
        accuracy = (correct_predictions / total_predictions) * 100
        dp = [[text, true_label, predicted_label, int(true_label == predicted_label)] for text, true_label, predicted_label in zip(texts, true_labels, predicted_labels)]
        self.save_to_excel(dp)
        self._accuracy = f"{accuracy:.2f}%"
        self.save_bin() # точность
        print(f"Процент совпадений: {self._accuracy}")
    
    def save_bin(self):
        data_to_save = {
        "model": self._model,
        "label_encoder": self._label_encoder,
        "report": self._report,
        "conf_matrix": self._conf_matrix,
        "execution_time": self._execution_time,
        "accuracy": self._accuracy
        }
        try:
            joblib.dump(data_to_save, self._model_path)
        except:
            print('Ошибка сохранения')
    
    def model_pipeline(self, X_train, y_train):
        pass

    def model_fit(self):
        self._model = self.model_pipeline(self._X_train, self._y_train)
        self.save_bin() # Сохранение модели
        y_pred = self._model.predict(self._X_test)
        self._report = classification_report(self._y_test, y_pred, target_names=self._label_encoder.classes_)
        self._conf_matrix = confusion_matrix(self._y_test, y_pred)
        self.save_bin() # Модель + матрица ошибок

        print(self._report)
        print("Confusion Matrix:")
        print(self._conf_matrix)
    
    def run(self):
        start_time = time.time()

        self.get_train_data()
        self.get_test_data()
        print('All data read')
        self.model_fit()
        self.testing()

        end_time = time.time()
        self._execution_time = end_time - start_time
        self.save_bin() # время выполнения
        print(f"Время выполнения: {self._execution_time} с.")