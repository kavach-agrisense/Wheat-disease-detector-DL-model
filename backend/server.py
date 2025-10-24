from fastapi import FastAPI, File, UploadFile
from .pred_helper import predict
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():
    return {"message": "Wheat disease detection model API is running 🚀"}


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path =f"/tmp/{file.filename}"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        prediction = predict(image_path)
        print(prediction)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


