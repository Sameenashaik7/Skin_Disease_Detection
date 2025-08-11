# ml/app.py
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from classify import predict
from PIL import Image
import uvicorn
import io

app = FastAPI(title="SkinNet Analyzer (Updated)")

@app.get("/")
def health():
    return {"status": "ok", "info": "SkinNet Analyzer (updated)"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        res = predict(pil_img, top_k=3)
        # return JSON with predictions and gradcam path
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/gradcam")
def get_gradcam(path: str):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(path, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("ml.app:app", host="0.0.0.0", port=7860, reload=True)
