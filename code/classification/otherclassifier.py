from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from plot import plot_confusion_matrix_and_roc, plot_multiclass_roc

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler(with_mean=False)
    svd = TruncatedSVD(n_components=3000)
    clf = make_pipeline(scaler, svd, SVC(gamma='auto'))
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],  
        'svc__gamma': ['scale', 'auto'],  
        'svc__kernel': ['linear', 'rbf', 'poly']  
    }
    # cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=5)
    # grid_search = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=cv_strategy, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best parameters found:")
    print(grid_search.best_params_)
    best_clf = grid_search.best_estimator_
    predictions = best_clf.predict(X_test)
    print("SVM evaluation with optimized parameters:")
    print(classification_report(y_test, predictions))
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("rf evaluation:")
    print(classification_report(y_test, predictions))
    # plot_confusion_matrix_and_roc(y_test=y_test,predictions=predictions)  
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
    

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    param_grid = {
        'xgbclassifier__n_estimators': [100, 200, 300],
        'xgbclassifier__max_depth': [3, 5, 7], 
        'xgbclassifier__colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    clf = Pipeline([
        ('xgbclassifier', xgb)
    ])
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=cv_strategy, verbose=2)

    grid_search.fit(X_train, y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)
    
    best_clf = grid_search.best_estimator_
    predictions = best_clf.predict(X_test)
    print("XGBoost evaluation with optimized parameters:")
    print(classification_report(y_test, predictions))
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

def train_and_evaluate_catboost(X_train, y_train, X_test, y_test):
    clf = CatBoostClassifier(iterations=100, silent=True)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("CatBoost evaluation:")
    print(classification_report(y_test, predictions))
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

def train_and_evaluate_stacking(X_train, y_train, X_test, y_test):
    scaler = StandardScaler(with_mean=False)
    estimators = [
        ('svm', make_pipeline(scaler, SVC(gamma='auto', probability=True))),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100)),
        ('catboost', CatBoostClassifier(iterations=100, silent=True))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Stacking evaluation:")
    print(classification_report(y_test, predictions))
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

def train_and_evaluate_majority_vote(X_train, y_train, X_test, y_test):
    scaler = StandardScaler(with_mean=False)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', make_pipeline(scaler, SVC(gamma='auto', probability=True))),
        ('xgb', XGBClassifier(n_estimators=100)),
        ('catboost', CatBoostClassifier(iterations=100, silent=True))
    ]
    clf = VotingClassifier(estimators=estimators, voting='soft')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Majority Vote evaluation:")
    print(classification_report(y_test, predictions))
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")