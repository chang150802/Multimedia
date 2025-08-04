# pip3 install flask opencv-python
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename, send_from_directory, send_file
import cv2
import os
from app import app
import uuid


ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['COMPRESSED_FOLDER'] = './static/compressed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'webp', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def processImage(filename, operation):
    print(f"the operation is {operation} and filename is {filename}")
    img = cv2.imread(f"static/uploads/{filename}")
    match operation:
        case "cgray":
            imgProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            newFilename = f"static/uploads/{filename}"
            cv2.imwrite(newFilename, imgProcessed)
            return newFilename
        case "cwebp":
            newFilename = f"static/uploads/{filename.split('.')[0]}.webp"
            cv2.imwrite(newFilename, img)
            return newFilename
        case "cjpg":
            newFilename = f"static/uploads/{filename.split('.')[0]}.jpg"
            cv2.imwrite(newFilename, img)
            return newFilename
        case "cpng":
            newFilename = f"static/uploads/{filename.split('.')[0]}.png"
            cv2.imwrite(newFilename, img)
            return newFilename
    pass
# chuyển đổi ảnh - done
@app.route("/edit", methods=["GET", "POST"])
def edit():
    if request.method == "POST":
        operation = request.form.get("operation")
        # kiểm tra xem yêu cầu bài viết có phần file không
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # Nếu người dùng không chọn tệp, trình duyệt sẽ gửi một
        # tập tin trống không có tên tập tin.
        if file.filename == '':
            flash('Chưa có file nào được chọn ')
            return "error no selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new = processImage(filename, operation)
            flash(f"Hình ảnh của bạn đã được xử lý và có  <a href='/{new}' target='_blank'>tại đây </a>")
            return render_template("convert.html")

    return render_template("convert.html")

#Rotate_done
#Rotate
@app.route('/rotate', methods=['GET', 'POST'])
def rotate_load():
    filename = None
    rotated_filename = request.args.get('rotated_filename')

    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f'Saved file: {file_path}')

    return render_template('rotate.html', filename=filename, rotated_filename=rotated_filename)


def rotate_image_cv(image_path, operation):
    img = cv2.imread(image_path)

    if operation == '90':
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif operation == '180':
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif operation == '270':
        rotated_img = cv2.rotate(img,  cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated_img = img  # No flip

    return rotated_img


@app.route('/rotate/<filename>/<operation>', methods=['GET', 'POST'])
def rotate(filename, operation):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f'Rotating image: {image_path} with operation: {operation}')
    rotated_img = rotate_image_cv(image_path, operation)

    rotated_filename = f'rotated_{filename}'
    rotated_img_path = os.path.join(app.config['UPLOAD_FOLDER'], rotated_filename)
    cv2.imwrite(rotated_img_path, rotated_img)
    print(f'Saved rotated image: {rotated_img_path}')

    return redirect(url_for('rotate_load', rotated_filename=rotated_filename))



#Flip_done
@app.route('/flip', methods=['GET', 'POST'])
def flip_load():
    filename = None
    flipped_filename = request.args.get('flipped_filename')

    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f'Saved file: {file_path}')

    return render_template('flip.html', filename=filename, flipped_filename=flipped_filename)


def flip_image_cv(image_path, operation):
    img = cv2.imread(image_path)

    if operation == 'vertical':
        flipped_img = cv2.flip(img, 0)
    elif operation == 'horizontal':
        flipped_img = cv2.flip(img, 1)
    elif operation == 'both':
        flipped_img = cv2.flip(img, -1)
    else:
        flipped_img = img  # No flip

    return flipped_img

#done
@app.route('/flip/<filename>/<operation>', methods=['GET', 'POST'])
def flip(filename, operation):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f'Flipping image: {image_path} with operation: {operation}')
    flipped_img = flip_image_cv(image_path, operation)

    flipped_filename = f'flipped_{filename}'
    flipped_img_path = os.path.join(app.config['UPLOAD_FOLDER'], flipped_filename)
    cv2.imwrite(flipped_img_path, flipped_img)
    print(f'Saved flipped image: {flipped_img_path}')

    return redirect(url_for('flip_load', flipped_filename=flipped_filename))



#remove background
def processImage_removebg(filename):

    img = cv2.imread(f"./static/uploads/{filename}")
    # First Convert to Grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, baseline = cv2.threshold(img_grey, 127, 255, cv2.THRESH_TRUNC)

    ret, background = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY)

    ret, foreground = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY_INV)

    foreground = cv2.bitwise_and(img, img, mask=foreground)  # Update foreground with bitwise_and to extract real foreground

    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Combine the background and foreground to obtain our final image
    new_filename=cv2.add(background, foreground)
    # new_filename = background + foreground
    NewFilename  = f"processed_{filename}"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],  NewFilename), new_filename)
    # NewFilename.save(os.path.join(app.config['UPLOAD_FOLDER'], NewFilename))
    # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], NewFilename), NewFilename)
    return NewFilename




#done
@app.route("/remove", methods=["GET", "POST"])
def removebg():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, browser submits an empty file without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If file is valid and allowed, save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and processed')

            # Process the image to remove background
            NewFilename = processImage_removebg(filename)
            if NewFilename is None:
                flash('Failed to process image')
                return redirect(request.url)

            return render_template('removebg.html', filename=filename, NewFilename=NewFilename)

    return render_template('removebg.html')


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

#Nén ảnh
# Function to compress the uploaded image based on quality parameter
@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)
def compress_image(filename, quality):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    dct_b = cv2.dct(np.float32(b))
    dct_g = cv2.dct(np.float32(g))
    dct_r = cv2.dct(np.float32(r))

    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    rows, cols = b.shape
    quantization_matrix = np.tile(quantization_matrix, (rows // 8 + 1, cols // 8 + 1))
    quantization_matrix = quantization_matrix[:rows, :cols]

    dct_quantized_b = np.round(dct_b / (quantization_matrix * (quality / 100.0)))
    dct_quantized_g = np.round(dct_g / (quantization_matrix * (quality / 100.0)))
    dct_quantized_r = np.round(dct_r / (quantization_matrix * (quality / 100.0)))

    dct_dequantized_b = dct_quantized_b * (quantization_matrix * (quality / 100.0))
    dct_dequantized_g = dct_quantized_g * (quantization_matrix * (quality / 100.0))
    dct_dequantized_r = dct_quantized_r * (quantization_matrix * (quality / 100.0))

    compressed_b = cv2.idct(dct_dequantized_b)
    compressed_g = cv2.idct(dct_dequantized_g)
    compressed_r = cv2.idct(dct_dequantized_r)

    compressed_b[compressed_b < 0] = 0
    compressed_g[compressed_g < 0] = 0
    compressed_r[compressed_r < 0] = 0
    compressed_b[compressed_b > 255] = 255
    compressed_g[compressed_g > 255] = 255
    compressed_r[compressed_r > 255] = 255

    compressed_b = np.uint8(compressed_b)
    compressed_g = np.uint8(compressed_g)
    compressed_r = np.uint8(compressed_r)

    compressed_img = cv2.merge((compressed_b, compressed_g, compressed_r))

    compressed_filename = f'{uuid.uuid4().hex}.jpg'
    cv2.imwrite(os.path.join(app.config['COMPRESSED_FOLDER'], compressed_filename), compressed_img)

    return compressed_filename
@app.route('/compress')
def index():
    return render_template('compress.html', filename=None, compress_image=None)

#Nén ảnh
class NewFilename:
    pass


@app.route("/compress", methods=['GET', 'POST'])
def compress_load():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        print(f'Saved file: {filename}')



        return render_template('compress.html', filename=filename ,NewFilename=NewFilename)
    return render_template('compress.html')
    # return redirect(request.url)

@app.route('/compress/<filename>/<operation>', methods=['GET', 'POST'])
def send_quantity_compress(filename , operation):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    quality = int(request.form['quality'])
    print(f'Flipping image: {filename} with operation: {quality}')
    compressed_filename = compress_image(filename, quality)
    return redirect(url_for('compress_load', compressed_filename=compressed_filename))
    # return render_template('compress.html', filename=filename, compress_image=compressed_filename)
app.run(debug=True, port=5001)


