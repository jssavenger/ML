import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Paths
_BASE_PATH= Path(__file__).parent.parent

# Models folder path
_MODEL_FOLDER_PATH= _BASE_PATH / "models"

# Model paths
scaler_path = _MODEL_FOLDER_PATH / "scaler.pkl"
model_path  =   _MODEL_FOLDER_PATH / "model.pkl"

# functions
from preprocessing import read_data, delete_column, split_data

def train_scaler(X_train, X_test, scaler):
    """Trains StandardScaler
            Args:
                X_train: X train dataset. (0.8)
                X_test : X test dataset   (0.2)
    """
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled 

def train_model_and_validations(X_train_scaled, y_train, X_test_scaled, y_test, model):
    """Trains LogisticRegression model
            Args:
                X_train_scaled: Scalied X train data.
                y_train: Targer values train dataset.
                X_test_scaled: Scalied X test data.
                y_test: Targer values test dataset.
                model: LogisticRegression model object.
    """
    # train model with X scaled dataset
    model.fit(X_train_scaled, y_train)

    # test predict
    y_pred = model.predict(X_test_scaled)

    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)

    # Valudations
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: %{accuracy_score(y_test, y_pred) * 100:.0f}")

def save_models(model_path: str, scaler_path: str, model, scaler):
    """Saves model with Pickle
            Args:
                model_path (str) : The Path where the model will save.
                scaler_path (str): The Scaler Path where the scaler will save.
                model: LogisticRegression model object.
                scaler: StandardScalar object. 
    """
    # save models
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        print("StandardScalar saved.")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        print("LogisticRegression model saved.")

if __name__ == "__main__":
    """Starts train.py
        Creates model and scaler object. Then starts train functions.
    """
    # create StandardScaler object
    scaler= StandardScaler()

    # create LogisticRegression object
    model = LogisticRegression(
        class_weight="balanced",
        penalty='l2',
        solver='sag',
        max_iter=1000
    )

    # Read data
    df = read_data()

    # Delete columns
    df = delete_column(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\n\n{X_train.columns}\n\n{X_train.info()}")
    # Train StandardScaler
    X_train_scaled, X_test_scaled  = train_scaler(X_train, X_test, scaler)

    # Tarin model and write validations
    train_model_and_validations(X_train_scaled, y_train, X_test_scaled, y_test, model)

    # Save models
    save_models(model_path, scaler_path, model, scaler)

    print(f"\nModel saved.\nLogistic Regression Path: {model_path}\nStandardScalar Path: {scaler_path}")