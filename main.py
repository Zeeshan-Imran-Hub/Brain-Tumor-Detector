import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import torchvision.transforms as transforms

app = FastAPI()

# Load the entire model
model = torch.load("C:/Users/ASUS/Desktop/Brain Tumor  Detection/Braintumor-detection/best_brain_tumor_model.pth",
                   map_location=torch.device('cpu'),
                   weights_only=False)

model.eval()

# Define the preprocessor pipeline
preprocessor = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict(image: Image.Image):
    image = preprocessor(image).unsqueeze(0)  # Preprocess the image and add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return 'Tumor Detected' if predicted.item() == 1 else 'No Tumor'

@app.get("/", response_class=HTMLResponse)
async def root():
    # Return a styled HTML page with a form to upload an image
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Detection</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                text-align: center;
                padding: 20px;
            }}
            h2 {{
                color: #007BFF;
            }}
            .upload-box {{
                background-color: #fff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                margin: 50px auto;
            }}
            input[type="file"] {{
                padding: 10px;
                margin-top: 20px;
            }}
            input[type="submit"] {{
                background-color: #007BFF;
                color: #fff;
                border: none;
                padding: 10px 20px;
                margin-top: 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }}
            input[type="submit"]:hover {{
                background-color: #0056b3;
            }}
            .result {{
                margin-top: 20px;
                font-size: 20px;
                color: #333;
            }}
            .footer {{
                margin-top: 50px;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="upload-box">
            <h2>Brain Tumor Detection</h2>
            <p>Please upload an MRI image to check for a brain tumor.</p>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <br>
                <input type="submit" value="Upload and Analyze">
            </form>
            {prediction_result}
        </div>
        <div class="footer">
            <p>&copy; 2024 Brain Tumor Detection. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content.format(prediction_result=""))

@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Make prediction
    prediction = predict(image)

    # Return the same UI with the prediction result displayed
    prediction_result = f'<div class="result">Prediction: <strong>{prediction}</strong></div>'
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Detection</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                color: #333;
                text-align: center;
                padding: 20px;
            }}
            h2 {{
                color: #007BFF;
            }}
            .upload-box {{
                background-color: #fff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                max-width: 500px;
                margin: 50px auto;
            }}
            input[type="file"] {{
                padding: 10px;
                margin-top: 20px;
            }}
            input[type="submit"] {{
                background-color: #007BFF;
                color: #fff;
                border: none;
                padding: 10px 20px;
                margin-top: 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }}
            input[type="submit"]:hover {{
                background-color: #0056b3;
            }}
            .result {{
                margin-top: 20px;
                font-size: 20px;
                color: #333;
            }}
            .footer {{
                margin-top: 50px;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="upload-box">
            <h2>Brain Tumor Detection</h2>
            <p>Please upload an MRI image to check for a brain tumor.</p>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <br>
                <input type="submit" value="Upload and Analyze">
            </form>
            {prediction_result}
        </div>
        <div class="footer">
            <p>&copy; 2024 Brain Tumor Detection. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content.format(prediction_result=prediction_result))

