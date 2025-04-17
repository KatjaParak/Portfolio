from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np


class ModelSelection:
    def __init__(self, df, feature):
        self.df = df
        self.X = df.drop(feature, axis=1)
        self.y = df[feature]
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.scaler = StandardScaler()
        self.normaliser = MinMaxScaler()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test, self.y_test, test_size=0.5, random_state=42)

        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val

    def feature_standardiser(self):
        self.split_data()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_val = self.scaler.transform(self.X_val)

        return self.X_train, self.X_test, self.X_val

    def feature_normaliser(self):
        self.feature_standardiser()

        self.norm_X_train = self.normaliser.fit_transform(self.X_train)
        self.norm_X_test = self.normaliser.transform(self.X_test)
        self.norm_X_val = self.normaliser.transform(self.X_val)

        return self.norm_X_train, self.norm_X_test, self.norm_X_val

    def classifier(self):
        self.split_data()
        self.feature_normaliser()

        estimators = [LogisticRegression(), KNeighborsClassifier(),
                      RandomForestClassifier()]
        param_grids = [{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga'], 'max_iter': [10000], 'penalty': ['elasticnet'],
                        'l1_ratio': [0.01, 0.1, 0.5, 1], 'dual': [False]},
                       {'n_neighbors': [1, 50, 100, 150, 175], 'metric': ['euclidean', 'manhattan', 'minkowski'],
                        'weights': ['uniform', 'distance'], 'algorithm': ['auto']},
                       {'n_estimators': [100, 200], 'max_depth': [None], 'criterion': ['gini', 'entropy'],
                        'min_samples_split': [10], 'max_features': [None], 'max_leaf_nodes': [None]}]

        models = []
        for estimator, param_grid in zip(estimators, param_grids):
            clf = GridSearchCV(estimator, param_grid,
                               cv=5, n_jobs=-1, verbose=0, scoring='recall')
            clf.fit(self.norm_X_train, self.y_train)
            models.append(clf)
        return models

    def get_best_score(self):
        models = self.classifier()

        best_scores = []
        for model in models:
            best_scores.append(model.best_score_)
            print(
                f"Recall metric for {model.estimator}: {model.best_score_:.3f}")

    def evaluate_model(self, model):
        self.split_data()
        self.feature_normaliser()

        model.fit(self.norm_X_train, self.y_train)
        y_pred_val = model.predict(self.norm_X_val)

        print(classification_report(self.y_val, y_pred_val,
              target_names=['absence', 'presence']))
        cm = confusion_matrix(self.y_val, y_pred_val)
        ConfusionMatrixDisplay(cm).plot()

    def voting_class(self):
        self.split_data()
        self.feature_normaliser()

        models = self.classifier()
        vote_clf = VotingClassifier(estimators=[
            ('lr', models[0]),
            ('knn', models[1]),
            ('rf', models[2])
        ], voting='hard')
        return vote_clf

    def test_model(self, model):
        self.split_data()
        self.feature_normaliser()

        X = np.concatenate((self.norm_X_train, self.norm_X_val))
        y = np.concatenate((self.y_train, self.y_val))

        model.fit(X, y)
        y_pred = model.predict(self.norm_X_test)

        print(classification_report(self.y_test, y_pred,
              target_names=['absence', 'presence']))
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
