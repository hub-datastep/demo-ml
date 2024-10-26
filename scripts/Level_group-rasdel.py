from sklearn.discriminant_analysis import StandardScaler
import base
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

class ModelRegres(base.ModelPipeline):
    def __init__(self) -> None:
        super().__init__()
        self._project = 'LevelGroup/ML Data'
        self._dir = 'Классификатор Разделов 4.2'

        self._model_path = "model/model_LevelGroup_Rasdel+3_NSI_from_01-10-2024.pkl"
        
        self._file_test_output_path = 'output_test/test_LevelGroup+3_Разделы_Материалы-НСИ-from-01-10-2024.xlsx'
        
        self._file_dataset_path = "c:\\Users\\Dmitry\\Downloads\\[LAST] Материалы НСИ from 01-10-2024 (для Даниила).xlsx"
        self._file_dataset_sheet = "main"
        self._file_dataset_column = ["name", "group"]

        self._file_test_input_path = "c:\\Users\\Dmitry\\Downloads\\[LAST] Материалы НСИ from 01-10-2024 (для Даниила).xlsx"
        self._file_test_input_sheet = "main"
        self._file_test_input_column = ["name", "group"]

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
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        return grid_search
    
m = ModelRegres()
m.run()