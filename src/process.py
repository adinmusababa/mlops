import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hydra
from omegaconf import DictConfig

# load raw data 
def load_raw_data(path: str) -> pd.DataFrame:
    """Load data mentah dari CSV."""
    df = pd.read_csv(path)
    print(f"[load] Shape awal: {df.shape}")
    return df

# hapus kolom yang tidak revelan dan tidak digunakan
def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Hapus kolom yang tidak digunakan sebagai fitur."""
    df = df.drop(columns=columns, errors="ignore")
    print(f"[drop] Kolom dihapus: {columns}")
    return df

# encodding data categorikal ke numerikal
def convert_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Konversi kolom ke tipe numerik.
    Nilai yang tidak bisa dikonversi (misal spasi kosong) → NaN → di-drop.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"[convert] '{col}': ditemukan {n_nan} nilai kosong (spasi) → di-drop")
            df = df.dropna(subset=[col])

    print(f"[convert] Shape setelah konversi: {df.shape}")
    return df

# encodingkan kolom binar Yes/No menjadi 0/1
def encode_binary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Encode kolom Yes/No menjadi 1/0."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"[encode_binary] Kolom diproses: {columns}")
    return df

# label encoderkan semua yang sudah didevinisikan
def label_encode_columns(
    df: pd.DataFrame,
    columns: list,
    encoders: dict = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label encode kolom kategorikal.
    - fit=True  → fit encoder baru (dipakai saat training)
    - fit=False → gunakan encoder yang sudah ada (dipakai saat inference)
    Mengembalikan (df, dict encoder per kolom).
    """
    if encoders is None:
        encoders = {}

    for col in columns:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    print(f"[label_encode] Kolom diproses: {columns}")
    return df, encoders

# Train split data
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pisahkan fitur dan target, lalu split train-test."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[split] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[split] Distribusi y_train:\n{y_train.value_counts()}")
    print(f"[split] Distribusi y_test:\n{y_test.value_counts()}")
    return X_train, X_test, y_train, y_test

# scale fitures numerik 
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list,
    scaler: StandardScaler = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale fitur numerik menggunakan StandardScaler.
    - fit=True  → fit scaler pada train set saja (hindari data leakage)
    - fit=False → gunakan scaler yang sudah ada
    """
    if fit:
        scaler = StandardScaler()
        X_train[columns] = scaler.fit_transform(X_train[columns])
    else:
        X_train[columns] = scaler.transform(X_train[columns])

    X_test[columns] = scaler.transform(X_test[columns])

    print(f"[scale] Kolom di-scale: {columns}")
    return X_train, X_test, scaler

# handling imbalance 

def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance sesuai strategi di config.
    - none        → tidak ada perubahan
    - smote       → oversample minority class
    - undersample → undersample majority class
    """
    if strategy == "none":
        print("[imbalance] Tidak ada penanganan imbalance.")
        return X_train, y_train

    elif strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "Package 'imbalanced-learn' belum terinstall. "
                "Jalankan: pip install imbalanced-learn"
            )
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[imbalance] SMOTE diterapkan. Distribusi baru:\n{y_train.value_counts()}")

    elif strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=random_state)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        print(f"[imbalance] Undersample diterapkan. Distribusi baru:\n{y_train.value_counts()}")

    else:
        raise ValueError(f"Strategi imbalance tidak dikenal: '{strategy}'")

    return X_train, y_train

# simpan hasil preprocessing
def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: DictConfig,
    scaler: StandardScaler,
    encoders: dict,
) -> None:
    """Simpan semua output ke folder data/processed/ dan models/."""
    for path in [config.data.x_train, config.data.x_test,
                 config.data.y_train, config.data.y_test]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    X_train.to_csv(config.data.x_train, index=False)
    X_test.to_csv(config.data.x_test, index=False)
    y_train.to_csv(config.data.y_train, index=False)
    y_test.to_csv(config.data.y_test, index=False)

    # Simpan scaler dan encoders ke models/ agar bisa dipakai saat inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/label_encoders.pkl")

    print("[save] Semua output berhasil disimpan:")
    print(f"  - {config.data.x_train}")
    print(f"  - {config.data.x_test}")
    print(f"  - {config.data.y_train}")
    print(f"  - {config.data.y_test}")
    print(f"  - models/scaler.pkl")
    print(f"  - models/label_encoders.pkl")

# main proses untuk menjalankan semua fungsi di atas secara berurutandef process_data(config: DictConfig) -> None:
@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig) -> None:
    print("=" * 50)
    print("Memulai preprocessing pipeline...")
    print("=" * 50)

    # Step 1: Load
    df = load_raw_data(config.data.raw)

    # Step 2: Drop kolom tidak relevan
    df = drop_columns(df, list(config.drop_columns))

    # Step 3: Fix tipe data
    df = convert_to_numeric(df, list(config.process.convert_to_numeric))

    # Step 4: Encode binary Yes/No
    df = encode_binary_columns(df, list(config.process.binary_columns))

    # Step 5: Label encode kategorikal
    df, encoders = label_encode_columns(
        df, list(config.process.label_encode_columns), fit=True
    )

    # Step 6: Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=config.target_column,
        test_size=config.process.test_size,
        random_state=config.process.random_state,
    )

    # Step 7: Scaling (fit pada train saja!)
    X_train, X_test, scaler = scale_features(
        X_train, X_test,
        columns=list(config.process.scale_columns),
        fit=True,
    )

    # Step 8: Handle imbalance
    X_train, y_train = handle_imbalance(
        X_train, y_train,
        strategy=config.process.imbalance_strategy,
        random_state=config.process.random_state,
    )

    # Step 9: Simpan semua output
    save_outputs(X_train, X_test, y_train, y_test, config, scaler, encoders)

    print("=" * 50)
    print("Preprocessing selesai!")
    print("=" * 50)


if __name__ == "__main__":
    process_data()
