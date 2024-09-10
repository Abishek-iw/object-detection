from flask import Flask, request, redirect, url_for, render_template_string, send_file
import os
import cv2
from ultralytics import YOLO
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
 
# Initialize YOLO model
model = YOLO("yolov8s-world.pt").to('cpu')  # or choose yolov8m/l-world.pt
 
# Define custom classes (only person)
model.set_classes(["person"])
 
# Set confidence threshold and NMS IoU threshold
model.conf = 0.2  # Adjusted confidence threshold
model.iou = 0.5   # Adjusted IoU threshold for NMS
 
# Define colors for each class
class_colors = {
    "person": (0, 255, 255),  # Yellow
}
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
def draw_boxes(image, results):
    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates (ensure integers)
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert to integers
            confidence = box.conf.item()  # Convert tensor to scalar
            class_id = box.cls.item()  # Convert tensor to scalar
 
            # Get class name and color
            class_name = model.names[int(class_id)]
            color = class_colors.get(class_name, (255, 0, 0))  # Default to red if not found
 
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
 
            # Draw the label and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    return image
 
@app.route('/', methods=['GET'])
def upload_form():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Upload Form</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label for="file">Choose an image to upload:</label>
                <input type="file" id="file" name="file" accept="image/*" required>
                <br><br>
                <input type="submit" value="Upload Image">
            </form>
        </body>
        </html>
    ''')
 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
   
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
   
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
       
        # Load the image
        image = cv2.imread(file_path)
       
        # Check if image loaded successfully
        if image is None:
            return "Error: Could not load image."
       
        # Execute prediction on the image
        results = model.predict(image)
       
        # Draw results on the image
        output_image = draw_boxes(image.copy(), results)  # Use a copy to avoid modifying original image
       
        # Save the processed image
        processed_filename = f"processed_{filename}"
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        cv2.imwrite(processed_file_path, output_image)
       
        return send_file(processed_file_path, mimetype='image/jpeg')
 
    return 'Invalid file format.'
 
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
 