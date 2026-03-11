"""Production API for Wind-Vision. Serve predictions via HTTP."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import os
from wind_vision.models.predict import predict_wind

app = FastAPI(
    title="Wind-Vision API",
    description="Predict wind speed from Lake Garda webcam images using ResNet-18",
    version="1.0.0"
)

# Temporary storage for uploaded images
UPLOAD_DIR = "data/tmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "online", "model": "resnet18_wind_regression"}

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload PNG or JPG.")

    # Save temp file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Run inference
        prediction = predict_wind(file_path)
        
        return {
            "filename": file.filename,
            "predicted_wind_kts": round(prediction, 1),
            "unit": "knots"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
