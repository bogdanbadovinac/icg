import sys
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16,  preprocess_input

# max_length=35
max_length=74

def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

# def predict_caption3(model, image_feature, tokenizer, max_length):
    
#     in_text = "startseq"
#     for i in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], max_length)

#         y_pred = model.predict([image_feature,sequence])
#         y_pred = np.argmax(y_pred)
        
#         word = idx_to_word(y_pred, tokenizer)
        
#         if word is None:
#             break
            
#         in_text+= " " + word
        
#         if word == 'endseq':
#             break
            
#     return in_text

def predict_caption2(model, image_feature, tokenizer, max_length):
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length)
            # predict next word
            ypred = model.predict([image_feature, sequence], verbose=0)
            # get index with high probability
            ypred = np.argmax(ypred)
            # convert index to word
            word = idx_to_word(ypred, tokenizer)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break
        
    return in_text

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Image Caption Generator'
        self.left = 200
        self.top = 200
        self.width = 1200
        self.height = 800
        self.vgg = 0
        self.image_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create "Browse" button
        self.browseButton = QPushButton('Browse', self)
        self.browseButton.setGeometry(550, 50, 150, 40)
        self.browseButton.clicked.connect(self.browse_image)

        # Create label for displaying selected image
        self.imageLabel = QLabel(self)
        self.imageLabel.setGeometry(50, 100, 300, 200)
        self.imageLabel.setAlignment(Qt.AlignCenter)

        # Create "Generate Caption" button
        self.generateButton = QPushButton('Generate Caption', self)
        self.generateButton.setGeometry(550, 720, 150, 40)
        self.generateButton.clicked.connect(self.generate_caption)

        self.show()

    def browse_image(self):
        # Open file dialog to select image
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter('Images (*.png *.xpm *.jpg *.bmp *.gif)')
        if file_dialog.exec_():
            self.image_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(self.image_path)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def generate_caption(self):
        
        
        image = load_img(self.image_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        
        if self.vgg == 1:
            model = load_model('second_model.h5')
            with open('tokenizer2.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            vgg_model = VGG16()
            vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output) 
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = vgg_model.predict(image, verbose=0)
            caption = predict_caption2(model, feature, tokenizer, max_length)
        else:
            print('TRYY')
            model = load_model('third_model.h5')
            with open('tokenizer3.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            dense_model = DenseNet201()
            dense_model = Model(inputs=dense_model.input, outputs=dense_model.layers[-2].output)
            image = image/255.
            image = np.expand_dims(image,axis=0)
            feature = dense_model.predict(image, verbose=0)
            caption = predict_caption2(model, feature, tokenizer, max_length)
            
        print(caption)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
