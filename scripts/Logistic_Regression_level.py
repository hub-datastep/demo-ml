from sklearn.discriminant_analysis import StandardScaler
import base
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

class ModelRegres(base.ModelPipeline):
    def __init__(self) -> None:
        super().__init__()
        self._model_path = "model/model_level_LogisticRegression.pkl"

        self._file_test_output_path = 'output_test/test_level_group_LogisticRegression.xlsx'

        self._file_dataset_path = "c:\\Users\\Dmitry\\Downloads\\LevelGroup_ training dataset (clear) (1).xlsx"
        self._file_dataset_column = ["name", "group"]
        self._file_dataset_sheet = "nomenclatures"

        self._file_test_input_path = "c:\\Users\\Dmitry\\Downloads\\Classifier_ levelgroup test cases.xlsx"
        self._file_test_input_sheet = "test-cases"
        self._file_test_input_column = ["Номенклатура поставщика", "Ожидание группа"]

        self._project = "LevelGroup/ML Data"
        self._dir = "week42"

    def model_pipeline(self, X_train, y_train):
        # Построение пайплайна с логистической регрессией
        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            LogisticRegression()
        )

        # Определение параметров для подбора
        param_grid = {
            'tfidfvectorizer__max_df': [0.85, 0.9, 0.95],  # Игнорировать слишком частые слова
            # 'tfidfvectorizer__min_df': [1, 2, 5],  # Игнорировать слишком редкие слова
            # 'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],  # Использовать униграммы, биграммы, триграммы
            # 'tfidfvectorizer__use_idf': [True, False],  # Применять ли IDF (обратная частота документа)
            # 'tfidfvectorizer__smooth_idf': [True, False],  # Сглаживание IDF
            # 'tfidfvectorizer__sublinear_tf': [True, False],  # Применять ли сублинейное масштабирование частотности

            'logisticregression__C': [0.01, 0.1, 1],  # Регуляризация
            # 'logisticregression__solver': ['liblinear', 'saga'],  # Использовать соответствующий решатель
            'logisticregression__max_iter': [100, 300, 500],  # Количество итераций
        }

        # Grid Search с 5-кратной кросс-валидацией
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )

        # Обучение модели
        grid_search.fit(X_train, y_train)
        return grid_search

    
m = ModelRegres()
m.run()