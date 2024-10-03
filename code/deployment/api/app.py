from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the trained model (dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)
# The Random Forest classifier was selected for the model due to its superior performance in comparison to other algorithms (https://archive.ics.uci.edu/dataset/109/wine)
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the input data schema 
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: int
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: int


@app.post("/predict")
def predict(input_data: WineInput):
    data = [[
        input_data.alcohol,
        input_data.malic_acid,
        input_data.ash,
        input_data.alcalinity_of_ash,
        input_data.magnesium,
        input_data.total_phenols,
        input_data.flavanoids,
        input_data.nonflavanoid_phenols,
        input_data.proanthocyanins,
        input_data.color_intensity,
        input_data.hue,
        input_data.od280_od315_of_diluted_wines,
        input_data.proline
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
