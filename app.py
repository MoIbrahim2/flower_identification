from fastapi import FastAPI, File, UploadFile
from gradio_client import Client, handle_file
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
app = FastAPI()


# Initialize the Gradio Client
flowerPredict19 = Client("releaf-nineteen/nineteen_Pi10")
flowerPredict100 = Client("releaf-nineteen/nineteen_releaf_")
diseasePredict = Client("releaf-nineteen/nineteen_PD")

@app.post("/predictFlower19")
async def predictFlower(image: UploadFile = File(...)):
    return await predict(flowerPredict19,"/predict",image=image)

@app.post("/predictFlower100")
async def predictFackness(image:UploadFile = File(...)):
    try:
        # Generate a unique temporary filename
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        
        # Save the uploaded file locally
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process the saved file with Gradio
        result = flowerPredict100.predict(
            image=handle_file(temp_filename),
            api_name= "/predict"
        )

        # Clean up the temporary file
        os.remove(temp_filename)

        # Return the prediction result
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predictDisease")
async def predict3(image:UploadFile = File(...)):
    return await predict(diseasePredict,"/predict",image=image)

async def predict(client,api_name,image: UploadFile = File(...)):
    try:
        # Generate a unique temporary filename
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        
        # Save the uploaded file locally
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process the saved file with Gradio
        result = client.predict(
            pil_image=handle_file(temp_filename),
            api_name= api_name
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