from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_models(cfg: dict) -> dict:
    m = cfg["models"]
    return {
        "logistic_regression": LogisticRegression(
            max_iter=m["logistic_regression"]["max_iter"],
            C=m["logistic_regression"]["C"],
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"],
            max_depth=m["random_forest"]["max_depth"],
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=m["xgboost"]["n_estimators"],
            max_depth=m["xgboost"]["max_depth"],
            learning_rate=m["xgboost"]["learning_rate"],
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=m["lightgbm"]["n_estimators"],
            max_depth=m["lightgbm"]["max_depth"],
            learning_rate=m["lightgbm"]["learning_rate"],
            class_weight="balanced",
            verbosity=-1,
        ),
        "svm": SVC(
            C=m["svm"]["C"],
            kernel=m["svm"]["kernel"],
            probability=True,
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=m["knn"]["n_neighbors"],
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=m["gradient_boosting"]["n_estimators"],
            max_depth=m["gradient_boosting"]["max_depth"],
            learning_rate=m["gradient_boosting"]["learning_rate"],
        ),
    }
