from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out
# Initialize the Flask application
app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'super_resolution_model_best.pth')
model = SRCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def extract_cb_cr(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cb, cr = cv2.split(ycbcr)
    return cb, cr

def convert_to_y_channel(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycbcr)
    return y

def process_image(input_image):
    input_image_Cb, input_image_Cr = extract_cb_cr(input_image)
    input_image_Y = convert_to_y_channel(input_image)

    input_tensor = torch.tensor(input_image_Y, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Add channel and batch dimensions and normalize

    GT_tensor = input_tensor.to('cpu')  # Assuming you're using CPU for inference

    # Perform inference
    with torch.no_grad():
        SR_tensor = model(GT_tensor)

    # Post-process SR tensor
    SR_tensor = SR_tensor.squeeze().cpu().numpy()  # Remove batch and channel dimensions and convert to NumPy
    SR_tensor = np.clip(SR_tensor * 255.0, 0, 255).astype(np.uint8)  # Rescale to 0-255 and convert to uint8

    # Combine SR Y channel with original Cb and Cr channels
    SR_ycbcr = cv2.merge((SR_tensor, input_image_Cb, input_image_Cr))
    SR_image = cv2.cvtColor(SR_ycbcr, cv2.COLOR_YCrCb2BGR)

    return SR_image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the image file
            input_image = Image.open(file.stream).convert('RGB')
            input_image = np.array(input_image)

            # Process the image
            output_image = process_image(input_image)

            # Save the processed image to a BytesIO object
            output_image_pil = Image.fromarray(output_image)
            output_img_io = io.BytesIO()
            output_image_pil.save(output_img_io, 'PNG')
            output_img_io.seek(0)

            # Save the processed image to disk for display and download
            output_image_path = os.path.join('static', 'output_image.png')
            output_image_pil.save(output_image_path)

            # Save the original input image to a BytesIO object
            input_image_pil = Image.fromarray(input_image)
            input_img_io = io.BytesIO()
            input_image_pil.save(input_img_io, 'PNG')
            input_img_io.seek(0)

            # Save the original image to disk for display
            input_image_path = os.path.join('static', 'input_image.png')
            input_image_pil.save(input_image_path)

            return render_template('upload.html', 
                                   input_image='input_image.png',  # Pass the correct filename
                                   output_image='output_image.png')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
