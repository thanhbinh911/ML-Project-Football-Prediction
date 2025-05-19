from src.feature_engineering import matchesB, predictors
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Train/test split
    X = matchesB[predictors]
    y = matchesB["target"]

    from sklearn.model_selection import (
        train_test_split,
        RandomizedSearchCV,
        StratifiedKFold,
    )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from scipy.stats import randint

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameter tuning for RandomForest
    param_dist = {
        "n_estimators": randint(100, 300),
        "max_depth": [None] + list(range(5, 31, 5)),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }

    rf = RandomForestClassifier(random_state=42)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv_strategy,
        n_jobs=-1,
        scoring="accuracy",
        verbose=2,
        random_state=42,
        return_train_score=True,
    )

    random_search.fit(X_train_val, y_train_val)
    print("Best Parameters found: ", random_search.best_params_)
    print("Best Cross-validation Score: {:.4f}".format(random_search.best_score_))
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    print(
        cv_results_df[
            [
                "param_n_estimators",
                "param_max_depth",
                "mean_train_score",
                "mean_test_score",
                "rank_test_score",
            ]
        ]
        .sort_values(by="rank_test_score")
        .head()
    )
    best_rf_model = random_search.best_estimator_
    y_pred_test = best_rf_model.predict(X_test)
    print("\n--- Test Set Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("Classification Report:\n", classification_report(y_test, y_pred_test))

    # Save the trained model
    import joblib

    joblib.dump(best_rf_model, "football_rf_model.pkl")
