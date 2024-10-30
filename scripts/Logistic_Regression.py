from sklearn.discriminant_analysis import StandardScaler
import base
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

class ModelRegres(base.ModelPipeline):
    def __init__(self) -> None:
        super().__init__()
        self._dir = 'fix_dataset+no_staircases+synthetic+synthetic'
        self._model_path = "model/model_unistroi_LogisticRegression+new+no_staircases+synthetic.pkl"

        self._file_dataset_path = "datasets/train_dataset_unistroi_new+no_staircases+synthetic.xlsx"
        self._file_dataset_sheet = 'main'
        self._file_dataset_column = ["noms","class"]

        self._file_test_output_path = 'output_test/test_model_unistroi_LogisticRegression+new+no_staircases+synthetic.xlsx'
        self._file_test_input_sheet = 'test-cases2'
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