from sklearn.ensemble import RandomForestClassifier
import training_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

class ModelRegres(training_model.ModelPipeline):
    def __init__(self) -> None:
        super().__init__()
        self._model_path = "model/model_unistroi_RandomForestClassifier.pkl"
    def model_pipeline(self, X_train, y_train):
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