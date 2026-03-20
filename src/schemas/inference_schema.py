from pydantic import BaseModel, Field

class Keys(BaseModel):
    Gender: int
    Age: int
    Population: int
    Number_of_Referrals: int = Field(alias="Number of Referrals")
    Tenure_in_Months: int = Field(alias="Tenure in Months")
    Avg_Monthly_Long_Distance_Charges: float = Field(alias="Avg Monthly Long Distance Charges")
    Internet_Service: int = Field(alias="Internet Service")
    Avg_Monthly_GB_Download: int = Field(alias="Avg Monthly GB Download")
    Premium_Tech_Support: int = Field(alias="Premium Tech Support")
    Paperless_Billing: int = Field(alias="Paperless Billing")
    Monthly_Charge: int = Field(alias="Monthly Charge")
    Total_Charges: int = Field(alias="Total Charges")
    Referred_a_Friend_Yes: bool = Field(alias="Referred a Friend_Yes")
    Phone_Service_Yes: bool = Field(alias="Phone Service_Yes")
    Multiple_Lines_Yes: bool = Field(alias="Multiple Lines_Yes")
    Internet_Type_DSL: bool = Field(alias="Internet Type_DSL")
    Internet_Type_Fiber_Optic: bool = Field(alias="Internet Type_Fiber Optic")
    Internet_Type_No_Internet: bool = Field(alias="Internet Type_No Internet")
    Unlimited_Data_Yes: bool = Field(alias="Unlimited Data_Yes")
    Contract_One_Year: bool = Field(alias="Contract_One Year")
    Contract_Two_Year: bool = Field(alias="Contract_Two Year")
    Payment_Method_Credit_Card: bool = Field(alias="Payment Method_Credit Card")
    Payment_Method_Mailed_Check: bool = Field(alias="Payment Method_Mailed Check")
