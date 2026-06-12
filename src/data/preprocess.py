import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    df = df.drop(columns=["molecule_id"])

    # encode categorical targets and drug_likeness feature
    le = LabelEncoder()
    if df[target].dtype == object:
        df[target] = le.fit_transform(df[target])

    if "drug_likeness" in df.columns and target != "drug_likeness":
        df["drug_likeness"] = (df["drug_likeness"] == "Yes").astype(int)

    if "toxicity_risk_level" in df.columns and target != "toxicity_risk_level":
        tox_map = {"Low": 0, "Medium": 1, "High": 2}
        df["toxicity_risk_level"] = df["toxicity_risk_level"].map(tox_map)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test, le, scaler
