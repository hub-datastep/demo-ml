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

        self._file_dataset_path = "datasets/LevelGroup_ fixed NSI groups by Dima.xlsx"
        self._file_dataset_column = ["name", "internal_group"]
        self._file_dataset_sheet = "Sheet0"

        self._file_test_input_path = "test-sets/Classifier_ levelgroup test cases from 13.09.2024 demo (made with GPT).xlsx"
        self._file_test_input_sheet = "test-cases"
        self._file_test_input_column = ["Номенклатура поставщика", "Ожидание номенклатура"]

        self._project = "LevelGroup/ML Data"
        self._dir = "week42"

    def model_pipeline(self, X_train, y_train):
        # Build pipeline with stacking
        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            LogisticRegression()
        )

        param_grid = {
            'logisticregression__C': [0.01, 0.1, 1, 10],
            'tfidfvectorizer__max_df': [0.9, 0.95],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        return grid_search
    
m = ModelRegres()
m.run()