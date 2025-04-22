from fastapi import FastAPI, File, UploadFile
from gradio_client import Client, handle_file
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

app = FastAPI()

# Initialize the Gradio Client
client = Client("releaf-nineteen/nineteen_f_hhhh")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Generate a unique temporary filename
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        
        # Save the uploaded file locally
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process the saved file with Gradio
        result = client.predict(
            pil_image=handle_file(temp_filename),
            api_name="/predict"
        )

        # Clean up the temporary file
        os.remove(temp_filename)

        # Return the prediction result
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)