"""
Fit/transform functions for training setup.

Used by the XGBoost, static FFNN and temporal-feature FFNN notebooks. Each
"_fit" function reads train data and returns parameters; each "_transform"
applies stored parameters without refitting.
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------
# Step 1: Winsorize (Tier 2 outliers)
# ---------------------------------------------------------------------

def winsorize_fit(train_df, feature_cols, low_pct=1, high_pct=99):
    """Return the 1st/99th percentile bounds for each feature, from train."""
    bounds = {}
    for col in feature_cols:
        low = train_df[col].quantile(low_pct / 100)
        high = train_df[col].quantile(high_pct / 100)
        bounds[col] = (low, high)
    return bounds


def winsorize_transform(df, bounds):
    """Clip each feature to its stored bounds. Returns a new DataFrame."""
    df = df.copy()
    for col, (low, high) in bounds.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df


# ---------------------------------------------------------------------
# Step 2: Fallback median imputation
# ---------------------------------------------------------------------

def impute_fit(train_df, feature_cols):
    """Return the median of each feature, from train."""
    medians = {col: train_df[col].median() for col in feature_cols}
    return medians


def impute_transform(df, medians):
    """Fill missing values with the stored medians. Returns a new DataFrame."""
    df = df.copy()
    for col, median_value in medians.items():
        df[col] = df[col].fillna(median_value)
    return df


# ---------------------------------------------------------------------
# Step 3: Feature scaling
# ---------------------------------------------------------------------

def scale_fit(train_df, feature_cols):
    """Fit a RobustScaler on train. Returns the fitted scaler."""
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def scale_transform(df, scaler, feature_cols):
    """Apply a fitted scaler to the feature columns. Returns a new DataFrame."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ---------------------------------------------------------------------
# Step 4: One-hot encoding
# ---------------------------------------------------------------------

def encode(df):
    """
    Encode gender and admission_type, dropping the original columns.

    Female and Emergency are the reference categories. Raises ValueError if an
    admission_type value appears in neither group list.
    """
    df = df.copy()

    df["gender_male"] = (df["gender"] == "M").astype(int)

    emergency_types = ["EW EMER.", "URGENT", "DIRECT EMER.",
                        "OBSERVATION ADMIT", "DIRECT OBSERVATION", "EU OBSERVATION"]
    elective_types = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]

    known = set(emergency_types) | set(elective_types)
    unexpected = set(df["admission_type"].unique()) - known
    if unexpected:
        raise ValueError(
            f"encode() found admission_type value(s) not in either group: {unexpected}. "
            "Add them to emergency_types or elective_types explicitly before proceeding."
        )

    df["admission_type_elective"] = df["admission_type"].isin(elective_types).astype(int)

    df = df.drop(columns=["gender", "admission_type"])
    return df


# ---------------------------------------------------------------------
# Step 5: SMOTE
# ---------------------------------------------------------------------

def smote_resample(X_train, y_train, random_state=42):
    """
    Generate synthetic minority-class samples (Chawla et al., 2002). Train data only.

    Returns the resampled X and y. imbalanced-learn is imported inside the
    function, so it is only needed when SMOTE is actually used.
    """
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# ---------------------------------------------------------------------
# Convenience: the whole fit/transform chain
# ---------------------------------------------------------------------

def fit_transform_pipeline(train_df, test_df, feature_cols, use_smote=False):
    """
    Run Winsorize, impute, scale and encode, fitting on train_df only.

    Applies SMOTE to train afterwards if use_smote is set. Returns the
    processed train_df, the processed test_df, and the fitted parameters.
    """
    bounds = winsorize_fit(train_df, feature_cols)
    train_df = winsorize_transform(train_df, bounds)
    test_df = winsorize_transform(test_df, bounds)

    medians = impute_fit(train_df, feature_cols)
    train_df = impute_transform(train_df, medians)
    test_df = impute_transform(test_df, medians)

    scaler = scale_fit(train_df, feature_cols)
    train_df = scale_transform(train_df, scaler, feature_cols)
    test_df = scale_transform(test_df, scaler, feature_cols)

    train_df = encode(train_df)
    test_df = encode(test_df)

    fitted_params = {"bounds": bounds, "medians": medians, "scaler": scaler}

    if use_smote:
        # train only; test_df is untouched
        X_train = train_df.drop(columns=["label"])
        y_train = train_df["label"]
        X_resampled, y_resampled = smote_resample(X_train, y_train)
        X_resampled["label"] = y_resampled
        train_df = X_resampled

    return train_df, test_df, fitted_params
