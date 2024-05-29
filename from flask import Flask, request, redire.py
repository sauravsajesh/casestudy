from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import cv2
import numpy as np
from PIL import Image
import svgwrite
import base64
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_line_drawing(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    line_art = 255 - dilated_edges
    cv2.imwrite(output_image_path, line_art)

def png_to_svg(input_path, output_path):
    with Image.open(input_path) as img:
        width, height = img.size
        img = img.convert('RGBA')
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        img_svg = dwg.image(href=f"data:image/png;base64,{img_base64}", size=(width, height))
        dwg.add(img_svg)
        dwg.save()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            line_drawing_path = os.path.join(app.config['UPLOAD_FOLDER'], 'line_drawing.png')
            svg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.svg')
            file.save(input_path)
            convert_to_line_drawing(input_path, line_drawing_path)
            png_to_svg(line_drawing_path, svg_path)
            return render_template('index.html', 
                                   original_image=url_for('uploaded_file', filename=filename),
                                   line_drawing=url_for('uploaded_file', filename='line_drawing.png'),
                                   svg_image=url_for('uploaded_file', filename='output.svg'))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
