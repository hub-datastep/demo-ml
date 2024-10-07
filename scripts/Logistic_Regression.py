from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import base
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

class ModelRegres(base.ModelPipeline):
    def __init__(self) -> None:
        super().__init__()
        self._model_path = "model/model_unistroi_LogisticRegression2.pkl"
        self._file_dataset_sheet = 'main'
        self._file_test_output_path = 'output_test/model_unistroi_LogisticRegression2.xlsx'
        self._file_dataset_path = "datasets/train_dataset_unistroi+sintetic.xlsx"
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