from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import io
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import APP_HOST, APP_PORT

import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Log environment variables
print("MONGODB_URL:", os.getenv("MONGODB_URL"))
print("MONGODB_URL_KEY:", os.getenv("MONGODB_URL_KEY"))

app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory="templates")

# Enable CORS for all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic Model for JSON input
class CustomerData(BaseModel):
    Age: int
    Education: int
    Marital_Status: int
    Parental_Status: int
    Children: int
    Income: float
    Total_Spending: float
    Days_as_Customer: int
    Recency: int
    Wines: int
    Fruits: int
    Meat: int
    Fish: int
    Sweets: int
    Gold: int
    Web: int
    Catalog: int
    Store: int
    Discount_Purchases: int
    Total_Promo: int
    NumWebVisitsMonth: int


# ✅ Train Model API
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(content={"status": True, "message": "Training successful!"})
    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


# ✅ Test Environment Variables API
@app.get("/test_env")
async def test_env():
    mongo_url = os.getenv("MONGODB_URL")
    return {"MONGODB_URL": mongo_url}


# ✅ Dashboard API
@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        # Provide some initial data for the dashboard
        # In a real scenario, these could be calculated from the DB or artifacts
        context = {
            "request": request,
            "total_customers": 2240,
            "avg_income": 52.2,
            "platinum_count": 450,
            "avg_spending": 605,
            "segments": ["Budget Customer", "Platinum Customer", "Moderate Spender"],
            "counts": [1100, 450, 690],
            "algo_labels": ["Random Forest", "XGBClassifier", "Logistic Regression", "SVM"],
            "algo_accuracy": [0.9710, 0.9665, 0.9554, 0.9174],
            "algo_precision": [0.9717, 0.9669, 0.9561, 0.9200],
            "algo_f1": [0.9711, 0.9666, 0.9555, 0.9178]
        }
        return templates.TemplateResponse("dashboard.html", context)
    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


# ✅ Render Customer Form (UI)
@app.get("/")
async def predictGetRouteClient(request: Request):
    try:
        return templates.TemplateResponse(
            "customer.html", {"request": request, "context": "Rendering"}
        )
    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


# ✅ Predict API (Form Input)
@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form_data = await request.form()
        # Convert form data to dict so Pydantic can parse and coerce types
        form_dict = {k: v for k, v in form_data.items()}
        data = CustomerData(**form_dict)

        print("Received data:", data.dict())

        # Convert JSON to list format expected by model
        input_data = [
            data.Age, data.Education, data.Marital_Status, data.Parental_Status, data.Children,
            data.Income, data.Total_Spending, data.Days_as_Customer, data.Recency, data.Wines,
            data.Fruits, data.Meat, data.Fish, data.Sweets, data.Gold, data.Web, data.Catalog,
            data.Store, data.Discount_Purchases, data.Total_Promo, data.NumWebVisitsMonth
        ]

        # Run prediction
        prediction_pipeline = PredictionPipeline()
        predicted_cluster = prediction_pipeline.run_pipeline(input_data=input_data)
        
        predicted_value = int(predicted_cluster[0])

        # Map cluster values to human-readable categories
        cluster_mapping = {
            0: "BUDGET CUSTOMER",
            1: "PLATINUM CUSTOMER",
            2: "MODERATE SPENDER"
        }
        category_name = cluster_mapping.get(predicted_value, f"CLUSTER {predicted_value}")
        
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "category": category_name}
        )

    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


@app.get("/predict_batch")
async def predict_batch_get(request: Request):
    try:
        return templates.TemplateResponse("batch.html", {"request": request})
    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        # Read uploaded file - use sep=None to auto-detect delimiters (comma, tab, etc.)
        df = pd.read_csv(file.file, sep=None, engine='python')
        
        # Run prediction
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict_batch(df)
        
        # Add predictions to dataframe
        df['predicted_cluster'] = predictions
        
        # Map cluster values to human-readable categories
        cluster_mapping = {
            0: "BUDGET CUSTOMER",
            1: "PLATINUM CUSTOMER",
            2: "MODERATE SPENDER"
        }
        df['category'] = df['predicted_cluster'].map(cluster_mapping)
        
        # Convert to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Save to a temporary file to return as response
        temp_file_path = "batch_predictions.csv"
        df.to_csv(temp_file_path, index=False)
        
        return FileResponse(
            path=temp_file_path, 
            filename="customer_segments.csv",
            media_type="text/csv"
        )

    except Exception as e:
        return JSONResponse(content={"status": False, "error": str(e)}, status_code=500)


# ✅ Run FastAPI Application
if __name__ == "__main__":
    import uvicorn

    print("MONGODB_URL:", os.getenv("MONGODB_URL"))
    print("MONGODB_URL_KEY:", os.getenv("MONGODB_URL_KEY"))


    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)
