from flask import Flask, render_template, request, jsonify
import base64
import os
import numpy as np
import pickle as pkl
from io import BytesIO
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
from keras_vggface.utils import preprocess_input
from mtcnn import MTCNN

features = np.array(pkl.load(open('features.pkl','rb')))
filenames = pkl.load(open('filenames.pkl','rb'))
model = load_model('restnet50.h5')

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return 'No file part'
        file = request.files['image_file']
        img_cv2 = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detector = MTCNN()
        results = detector.detect_faces(img_cv2)

        x,y,width,height = results[0]['box']
        face = img_cv2[y:y+height,x:x+width]

        image = Image.fromarray(face)
        image = image.resize((224,224))
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array,axis=0)
        preprocessed_img = preprocess_input(expanded_img)

        result = model.predict(preprocessed_img).flatten()
        #print(result)
        similarity = []
        for i in range(len(features)):
            similarity.append(cosine_similarity(result.reshape(1,-1),features[i].reshape(1,-1))[0][0])

        index = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

        #print(filenames[index])
        path = filenames[index]
        filename = os.path.basename(path)
        name_without_extension = os.path.splitext(filename)[0]
        name = name_without_extension.split('\\')[-1]
        #print(name)

        final = cv2.imread(filenames[index])
        _, img_encoded = cv2.imencode('.jpg', final)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        return render_template('index.html', image=img_base64,prediction_text= 'You look like {}' . format(name) ) 
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)