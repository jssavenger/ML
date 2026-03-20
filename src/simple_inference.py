import pickle
import pandas as pd
from pathlib import Path

# Paths
_BASE_PATH= Path(__file__).parent.parent

# Models folder path
_MODEL_FOLDER_PATH= _BASE_PATH / "models"

# Model paths
scaler_path = _MODEL_FOLDER_PATH / "scaler.pkl"
model_path  =   _MODEL_FOLDER_PATH / "model.pkl"

def load_models(path: Path, name: str = "LoadedModel"):
    """Reads models
            Args:
                path (Path): The Model where will be load.
    """
    with open(path, "rb") as f:
        name = pickle.load(f)
    
    return name

new_data = {
    "Gender": 1,
    "Age": 23,
    "Population": 55668,
    "Number of Referrals": 1,
    "Tenure in Months": 2,
    "Avg Monthly Long Distance Charges": 44.07,
    "Internet Service": 1,
    "Avg Monthly GB Download": 6,
    "Premium Tech Support": 0,
    "Paperless Billing": 0,
    "Monthly Charge": 65,
    "Total Charges": 122,
    "Referred a Friend_Yes": True,
    "Phone Service_Yes": True,
    "Multiple Lines_Yes": False,
    "Internet Type_DSL": False,
    "Internet Type_Fiber Optic": True,
    "Internet Type_No Internet": False,
    "Unlimited Data_Yes": False,
    "Contract_One Year" : True,
    "Contract_Two Year" : False,
    "Payment Method_Credit Card": True,
    "Payment Method_Mailed Check": False
}



if __name__ == "__main__":
    """Starts inference for Logistic Regression model on the data the never model seen.
    """
    # Upload Logistic Regression model and StandardScalar
    scaler = load_models(scaler_path)
    model  = load_models(model_path)

    print("Logistic Regression model and StandardScalar uploaded.")

    new_df = pd.DataFrame([new_data])
    print(f"\n{new_df}\n")

    result = model.predict(new_df)

    print("Model Response: ", result, type(result))