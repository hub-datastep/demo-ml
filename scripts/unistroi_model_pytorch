import time
import joblib
import numpy as np
import pandas as pd
from google_uploader import GoogleUploader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class ModelPipeline:
    def __init__(self) -> None:
        self._project = 'Унистрой'
        self._dir      = 'test_pytorch'

        self._file_dataset_path = f"datasets/ynistroi.xlsx"
        self._file_dataset_sheet = "Лист1"
        self._file_dataset_column = ["name", "group"]
        
        self._file_test_input_path = f"test-sets/Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx"
        self._file_test_input_sheet = "test-cases"
        self._file_test_input_column = ["Номенклатура поставщика", "Ожидание группа"]

        self._file_test_output_path = f"output_test/Classifier_test-case_unistroi.xlsx"
        self._file_test_output_column = ['NOMs','GROUP', 'AI', 'T|F']

        self._model_path = "model/model_unistroi.pth"

        self._model = None
        self._vectorizer = None
        self._label_to_index = None
        self._index_to_label = None
        self._num_classes = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._report = None
        self._conf_matrix = None
        self._execution_time = None
        self._accuracy = None

        self._X_train_texts = None
        self._X_test_texts = None
        self._y_train = None
        self._y_test = None
        self._texts = None
        self._true_labels = None
        self._model_path = "model/model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa.pkl"
        self._file_dataset_sheet = 'main'
        self._file_test_output_path = 'output_test/test_model_unistroi_LogisticRegression+syntetics+clear_big_class+not_lestnitsa.xlsx'
        self._file_dataset_path = "datasets\\train_dataset_unistroi+syntetics+clear_big_class+not_lestnitsa.xlsx"
        self._file_test_input_sheet = 'test-cases2'

    def __init_google_drive(self):
        self.drive = GoogleUploader(parent_folder_path=self._project, new_folder_name=self._dir, files=[self._file_dataset_path, self._file_test_input_path,self._file_test_output_path, self._model_path])
    
    def __upload_to_google_drive(self):
        print(self.drive.upload_files_to_path())
    
    def read_excel_columns(self, file_path, columns, sheet):
        try:
            df = pd.read_excel(file_path, usecols=columns, sheet_name=sheet)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
            return None, None
        except ValueError as e:
            print(f"Ошибка: Некорректные данные в файле или неверные колонки. {e}")
            return None, None
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return None, None

        try:
            noms = df[columns[0]].astype(str).tolist()
            groups = df[columns[1]].astype(str).tolist()
        except Exception as e:
            print(f"Ошибка при преобразовании данных: {e}")
            return None, None

        return noms, groups 

    def get_train_data(self):
        # Считывание из Excel
        X, y = self.read_excel_columns(self._file_dataset_path, self._file_dataset_column, self._file_dataset_sheet)
        if X is None or y is None:
            return

        # Создание словаря меток
        classes = sorted(list(set(y)))
        self._label_to_index = {label: idx for idx, label in enumerate(classes)}
        self._index_to_label = {idx: label for idx, label in enumerate(classes)}
        self._num_classes = len(classes)

        # Кодирование меток
        y_encoded = np.array([self._label_to_index[label] for label in y])

        # Разделение данных на обучающую и тестовую выборки
        self._X_train_texts, self._X_test_texts, self._y_train, self._y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        # Инициализация TF-IDF Vectorizer
        self._vectorizer = TfidfVectorizer(max_df=0.95)
        self._X_train_features = self._vectorizer.fit_transform(self._X_train_texts)
        self._X_test_features = self._vectorizer.transform(self._X_test_texts)

    def get_test_data(self):
        self._texts, self._true_labels = self.read_excel_columns(
            self._file_test_input_path, self._file_test_input_column, self._file_test_input_sheet
        )
        if self._texts is not None:
            self._texts_features = self._vectorizer.transform(self._texts)
    
    def save_to_excel(self, data):
        try:
            df = pd.DataFrame(data, columns=self._file_test_output_column)
            df.to_excel(self._file_test_output_path, index=False)
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def save_bin(self):
        data_to_save = {
            "model_state_dict": self._model.state_dict(),
            "vectorizer": self._vectorizer,
            "label_to_index": self._label_to_index,
            "index_to_label": self._index_to_label,
            "report": self._report,
            "conf_matrix": self._conf_matrix,
            "execution_time": self._execution_time,
            "accuracy": self._accuracy
        }
        try:
            torch.save(data_to_save, self._model_path)
        except Exception as e:
            print(f'Ошибка сохранения: {e}')

    class TorchDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features.toarray(), dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            outputs = self.linear(x)
            return outputs

    def model_fit(self):
        if self._X_train_texts is None:
            print("Данные для обучения отсутствуют.")
            return

        # Создание датасетов и загрузчиков
        train_dataset = self.TorchDataset(self._X_train_features, self._y_train)
        test_dataset = self.TorchDataset(self._X_test_features, self._y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Инициализация модели
        input_dim = self._X_train_features.shape[1]
        num_classes = self._num_classes
        model = self.LogisticRegressionModel(input_dim, num_classes).to(self._device)

        # Вычисление весов классов для функции потерь
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self._y_train),
            y=self._y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(self._device), labels.to(self._device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Эпоха {epoch + 1}/{num_epochs}, Потеря: {total_loss / len(train_loader)}")

        self._model = model

        # Оценка модели
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self._device), labels.to(self._device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        target_names = [self._index_to_label[i] for i in range(self._num_classes)]
        self._report = classification_report(all_labels, all_preds, target_names=target_names)
        self._conf_matrix = confusion_matrix(all_labels, all_preds)
        self.save_bin()  # Сохранение модели и отчетов

        print(self._report)
        print("Confusion Matrix:")
        print(self._conf_matrix)

    def testing(self):
        if self._model is None:
            print("Модель не обучена.")
            return
        if self._texts is None:
            print("Тестовые данные отсутствуют.")
            return

        model = self._model
        model.eval()
        features = torch.tensor(self._texts_features.toarray(), dtype=torch.float32).to(self._device)

        with torch.no_grad():
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predicted_labels = [self._index_to_label[pred.item()] for pred in preds]

        # Сравнение предсказанных меток с истинными
        correct_predictions = sum(1 for true, pred in zip(self._true_labels, predicted_labels) if true == pred)
        total_predictions = len(self._texts)
        accuracy = (correct_predictions / total_predictions) * 100

        dp = [[text, true_label, predicted_label, int(true_label == predicted_label)] 
              for text, true_label, predicted_label in zip(self._texts, self._true_labels, predicted_labels)]
        self.save_to_excel(dp)
        self._accuracy = f"{accuracy:.2f}%"
        self.save_bin()  # точность
        print(f"Процент совпадений: {self._accuracy}")

    def run(self):
        self.__init_google_drive()
        start_time = time.time()

        self.get_train_data()
        self.get_test_data()
        print('All data read')
        self.model_fit()
        self.testing()

        end_time = time.time()
        self._execution_time = end_time - start_time
        self.save_bin()  # время выполнения
        self.__upload_to_google_drive()
        print(f"Время выполнения: {self._execution_time} с.")

# Запуск
if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run()
