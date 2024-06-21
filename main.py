import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes

dataset=pd.read_csv('./Data/dataset.csv',compression='gzip')

app = FastAPI()

class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False

class PredictionIn(BaseModel):
    nutrition_input: List[float]
    ingredients: List[str] = []
    params: Optional[Params] = Params()

class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
def home():
    return {"health_check": "OK"}
@app.post("/predict/")
def update_item(prediction_input: PredictionIn):
    try:
        print("Starting recommendation process")
        recommendation_dataframe = recommend(
            dataset,
            prediction_input.nutrition_input,
            prediction_input.ingredients,
            prediction_input.params.dict()
        )
        print("Recommendation process completed")

        if recommendation_dataframe is None:
            print("No recommendations found")
            return {"output": None}
        
        return recommendation_dataframe.to_json()

    except Exception as err:
        print("An error occurred:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(err))