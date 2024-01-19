from flask import Flask,render_template,request,Response,send_from_directory,redirect
from PIL import Image
import cv2
import numpy as np
import os
import datetime, time
from threading import Thread
import shutil
from encodeAndDecodeImage import encode_image,decode_image
from encodeAndDecodeVideo import encode_text,decode_text


app=Flask(__name__)

# Save Image Formater
ALLOWED_EXTENSIONS_Images = {'png', 'jpg', 'jpeg'}

#make  directory to save picture and videos
directory_list=['./static/encode_Images','./static/encode_Videos','./static/images','./static/images/Download','./static/images/decodeImages','./static/Real_Time_Take_Images','./static/Real_Time_Take_Videos']

for i in directory_list:
    try:
        os.mkdir(i)
    except OSError as error:
        pass

# Save Images Directory variable
save_directory = './static/encode_Images/'

# Specify directory to download from . . .
DOWNLOAD_DIRECTORY = "./static/images/Download/"

# Save Videos Variables
app.config['UPLOAD_FOLDER'] = './static/encode_Videos/'
app.config['ALLOWED_EXTENSIONS_Videos'] = {'mp4', 'avi', 'mkv', 'mov'}  # Add more file extensions as needed

accept_image={'png'}
def allowed_file_accept_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in accept_image

# Allow Videos File Allow Function
def allowed_file_Videos(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS_Videos']


# Home Page for steganography Project
@app.route("/")
def Home():
    return render_template("home.html")

# About Page for steganography Project
@app.route("/about")
def about():
    return render_template("about.html")

# Allow Images File Function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_Images

################################################################
########## Take Real Time Image and Videos  #########################
#########################################################

global capture,rec_frame, grey, switch, neg, face, rec, out
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
global imageName
imageName=None


#Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

camera = cv2.VideoCapture(0)

# Get file Name From a Directory Function
def get_file_names(directory_path):
    file_names = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the current item is a file (not a subdirectory)
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_names.append(filename)

    return file_names


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame

def copy_image(source_folder, destination_folder, image_name):
    # Construct the full paths for the source and destination images
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(destination_folder, image_name)

    try:
        # Copy the image
        shutil.copy2(source_path, destination_path)
        print(f"Image '{image_name}' copied successfully from {source_folder} to {destination_folder}.")
    except FileNotFoundError:
        print(f"Error: Image '{image_name}' not found in {source_folder}.")
    except PermissionError:
        print(f"Error: Permission denied. Check if you have the required permissions.")


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                filename = "shot_{}.png".format(str(now).replace(":", ''))
                p = os.path.sep.join(['./static/Real_Time_Take_Images/', filename])
                cv2.imwrite(p, frame)
            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/videoFrame')
def index():
    return render_template('videoFrame.html',framename=imageName)

@app.route('/imageFrame')
def indexImage():
    return render_template('imageFrame.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image_feed')
def image_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('./static/Real_Time_Take_Videos/vid_{}.mp4'.format(str(now).replace(":", '')),
                                      fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()


    elif request.method == 'GET':
        return render_template('videoFrame.html',framename=imageName)
    return render_template('videoFrame.html',framename=imageName)




@app.route('/imageRequests', methods=['POST', 'GET'])
def tasksImage():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1

    elif request.method == 'GET':
        return render_template('imageFrame.html')
    return render_template('imageFrame.html')



#############################################################

@app.route('/shareFile')
def shareFile():
    return render_template('shareFile.html')

@app.route('/shareFile',methods=['POST'])
def shareFileRequest():
    sharefile = request.form['sendprocess']
    return render_template('shareFile.html',sendprocess=sharefile)

@app.route("/localImageDecode")
def localImageDecode():
    return render_template("decodeImage.html")


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']
    img=Image.open(file.stream)
    return "not ready yet!!!"


@app.route('/localImage')
def localImage():
    return render_template('localImage.html')

###################################################################
################### Encode Images and Videos Functions#############
@app.route('/localImage',methods=['POST'])
def encode_Upload_file():
    file= request.files['imagefile']
    text = request.form['secretData']
    if file and allowed_file_Videos(file.filename) and text:
        save_path_video=app.config['UPLOAD_FOLDER']+file.filename
        # Save the file to the specified upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        filename = os.path.splitext(file.filename)[0] + '.png'
        save_video_image = DOWNLOAD_DIRECTORY + filename
        encode_text(save_path_video,text,save_video_image)
        return render_template('localImage.html', encodeVideo=filename)

    if file and allowed_file(file.filename) and text:
        save_path_image = save_directory + file.filename
        # Save the image to the specified directory
        file.save(os.path.join(save_directory, file.filename))
        if file and allowed_file_accept_image(file.filename):
            encode_image(save_path_image, text, file.filename)
            return render_template('localImage.html', encodeImage=file.filename)
        filename = os.path.splitext(file.filename)[0] + '.png'
        encode_image(save_path_image,text,filename)
        return render_template('localImage.html',encodeImage= filename)
    return "<h1 style='color:blue;'>Enter All Information Properly!!!!! </h1>"


@app.route('/get-files/<path:path>',methods = ['GET','POST'])
def get_files(path):

    """Download a file."""
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, path, as_attachment=True)
    except FileNotFoundError:
        return render_template('404.html'),404


##################################################################################
############################# Decode Images and Videos Function ##################
###################################################################################

@app.route('/decodeuploadfile',methods = ['GET','POST'])
def decode_Upload_file():
    file = request.files['imagefile']
    value = request.form['decodeprocess']
    result=''
    if value=='Image Decoding' and file:
        save_path_image = "./static/images/decodeImages/" + file.filename
        file.save(os.path.join('./static/images/decodeImages/', file.filename))
        result = decode_image(save_path_image)
    elif value=='Video Decoding' and file:
        save_path_image = "./static/images/decodeImages/" + file.filename
        file.save(os.path.join('./static/images/decodeImages/', file.filename))
        result = decode_text(save_path_image)
    else:
        return "<h1 style='color:blue;'>Enter All Information Properly!!!!!</h1>"

    return render_template('localDecode.html',result_vlaue=result)



##################  Download Images/Videos ######################

@app.route("/downloadFileImage")
def downloadFileImage():
    filenames = get_file_names('./static/Real_Time_Take_Images/')
    filenames = filenames[::-1]

    return render_template("download_File_Image.html",file_lists_names=filenames)


@app.route("/downloadFileVideo")
def downloadFileVideo():
    filenames = get_file_names('./static/Real_Time_Take_Videos/')
    filenames = filenames[::-1]

    return render_template("download_File_Videos.html",file_lists_names=filenames)

@app.route('/get-files-images/<path:path>',methods = ['GET','POST'])
def get_files_images(path):

    """Download a file."""
    try:
        return send_from_directory("./static/Real_Time_Take_Images/", path, as_attachment=True)
    except FileNotFoundError:
        return render_template('404.html'),404

@app.route('/get-files-videos/<path:path>',methods = ['GET','POST'])
def get_files_videos(path):

    """Download a file."""
    try:
        return send_from_directory("./static/Real_Time_Take_Videos/", path, as_attachment=True)
    except FileNotFoundError:
        return render_template('404.html'),404


###################################################################################

@app.route("/localDecode")
def localVideo():
    return render_template('localDecode.html')




# Create Custom Error Page
# Invalid URL
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


# Internal Server Error
@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500

# Run the code 
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)

camera.release()
cv2.destroyAllWindows()



