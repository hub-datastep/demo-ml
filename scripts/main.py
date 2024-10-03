import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def model_fit(X_train, X_test, y_train, y_test):
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Build pipeline with stacking
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )

    # Set up parameter grid for GridSearchCV
    param_grid = {
        'tfidfvectorizer__ngram_range': [(1,1), (1,2)],
        'tfidfvectorizer__max_df': [0.85, 0.9, 1.0],
        'tfidfvectorizer__min_df': [1, 5],
    }

    # Define GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='precision_macro',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train_enc)

    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test_enc, y_pred, target_names=le.classes_)
    conf_matrix = confusion_matrix(y_test_enc, y_pred)

    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    return grid_search, le, report, conf_matrix

def get_train_data():
    sheets = pd.read_excel('ynistroi.xlsx', sheet_name='test-cases')

    arrays = {sheet_name: pd.DataFrame(df) for sheet_name, df in sheets.items()}

    first_sheet_name = list(arrays.keys())[0]
    training_data_df = arrays[first_sheet_name]
 
    selected_columns = ["Номенклатура поставщика",	"Ожидание группа"]
    training_data_df = training_data_df[selected_columns]
    # Пример данных
    data = np.array(training_data_df)
    # Разделение данных на номенклатуру и классы
    X = data[:, 0]  # Номенклатура
    y = data[:, 1]  # Классы

    # Разделение данных на обучающую и тестовую выборки
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def read_file() -> dict[str, str]:
    answer = []
    # Чтение файла Excel
    file_path = f'Classifier_ unistroy UTDs test-cases (fix)_30.09.2024.xlsx'

    # Чтение всех листов в книге Excel в виде словаря {имя листа: DataFrame}
    sheets = pd.read_excel(file_path, sheet_name=None)

    # Преобразование данных каждого листа в двумерные массивы (списки списков)
    arrays = {sheet_name: df.values.tolist() for sheet_name, df in sheets.items()}

    # Пример доступа к массиву данных первого листа
    first_sheet_name = list(arrays.keys())[0]
    first_sheet_array = arrays[first_sheet_name]

    for row in first_sheet_array:
        answer.append([row[1], row[2]])
    print(len(answer))
    return answer

def test_model(unseen_data, model, label_encoder):
    texts = [item[0] for item in unseen_data]
    true_labels = [item[1] for item in unseen_data]

    # Предсказание на новых данных
    predicted_labels_encoded = model.predict(texts)
    
    # Преобразование числовых меток обратно в строковые с помощью LabelEncoder
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
    
    # Сравнение предсказанных меток с истинными
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total_predictions = len(unseen_data)
    
    # Вычисляем процент совпадений
    accuracy = (correct_predictions / total_predictions) * 100
    dp = [[text, true_label, predicted_label] for text, true_label, predicted_label in zip(texts, true_labels, predicted_labels)]
    save_to_excel(dp, ['NOMs','GROUP', 'II'])
    print(f"Процент совпадений: {accuracy:.2f}%")
    return f"{accuracy:.2f}%"

def main():
    start_time = time.time()
    
    model_path = f"model_ynistroi_007.pkl"
    X_train, X_test, y_train, y_test = get_train_data()
    model, label_encoder, report, conf_matrix = model_fit(X_train, X_test, y_train, y_test)
    end_time = time.time()
    execution_time = end_time - start_time
    accuracy = test_model(read_file(), model, label_encoder)
    data_to_save = {
        "model": model,
        "label_encoder": label_encoder,
        "report": report,
        "conf_matrix": conf_matrix,
        "execution_time": execution_time,
        "accuracy": accuracy
    }
    
    joblib.dump(data_to_save, model_path)
    
def save_to_excel(data, column_names, file_name='Classifier_test-case_unistroi.xlsx'):
    try:
        # Преобразуем массив в DataFrame
        df = pd.DataFrame(data)
        
        # Сохраняем DataFrame в Excel файл
        df = pd.DataFrame(data, columns=column_names)
        df.to_excel(file_name, index=False)
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Время выполнения: {execution_time} секунд")

