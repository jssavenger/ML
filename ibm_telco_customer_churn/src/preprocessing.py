import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
_BASE_PATH= Path(__file__).parent.parent
_DATA_PATH= _BASE_PATH / "data" / "raw" / "telco.csv"

# List to be deleted
delete_columns = [
    "Churn Score",
    "Customer Status",
    "City",
    "Longitude",
    "Offer",
    "Quarter",
    "Zip Code",
    "Latitude",
    "State",
    "Country",
    "Customer ID",
    "Number of Dependents",
    "Dependents",
    "Under 30",
    "Senior Citizen",
    "Married",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "Satisfaction Score",
    "Churn Reason",
    "Churn Category",
    "CLTV"
    ]

# Read to the dataset
def read_data():
    """This function reads dataset.
    """
    df = pd.read_csv(_DATA_PATH)
    print("Datased readed.")
    return df

def delete_column(df):
    """This function deletes columns on the dataset.
    """
    # drop this columns 
    df = df.drop(delete_columns, axis=1, inplace=False)

    # turn to numeric
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # none sutunlarini 0 ile doldur
    df['Total Charges'].fillna(0, inplace=True)

    df['Internet Type'].fillna("No Internet",inplace=True)

    df['Churn Label']= df['Churn Label'].replace("Yes", 1)
    df['Churn Label']= df['Churn Label'].replace("No", 0)

    df['Internet Service']= df['Internet Service'].replace("Yes", 1)
    df['Internet Service']= df['Internet Service'].replace("No", 0)

    df['Gender']= df['Gender'].replace("Male", 1)
    df['Gender']= df['Gender'].replace("Female", 0)

    df['Paperless Billing']= df['Paperless Billing'].replace("Yes", 1)
    df['Paperless Billing']= df['Paperless Billing'].replace("No", 0)

    df['Premium Tech Support']= df['Premium Tech Support'].replace("Yes", 1)
    df['Premium Tech Support']= df['Premium Tech Support'].replace("No", 0)

    df= pd.get_dummies(df, drop_first=True)

    return df

def split_data(df):
    """Splits train and split
    """
    X = df.drop(columns='Churn Label')
    y = df['Churn Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    print(f"X_train: {len(X_train)} | X_test: {len(X_test)} | {len(y_train)} | {len(y_test)}")

    return X_train, X_test, y_train, y_test





    

