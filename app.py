from flask import Flask, render_template, Response
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
import sounddevice as sd
import soundfile
import tempfile
import librosa
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

class_names_video = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
class_names_audio = ['Fear', 'Pleasant_surprise', 'Sad', 'angry', 'disgust', 'happy', 'neutral','angry','disgust','fear','happy','neutral','pleasant_surprised','sad']

# Load video emotion classification model
model_video = timm.create_model('efficientnet_b2', pretrained=True)
model_video.load_state_dict(torch.load('best_model_epoch_7.pt', map_location=torch.device('cpu')))
model_video.eval()

# Load audio emotion classification model
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.25)
        self.lstm1 = nn.LSTM(64, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

model_audio = CNN_LSTM(180, 32, 14)
model_audio.load_state_dict(torch.load('speech_classification_model.pth'))
model_audio.eval()

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_emotion_video(frame):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = test_transform(frame_pil).unsqueeze(0)
    with torch.no_grad():
        output = model_video(input_image)
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_class_name = class_names_video[predicted_class]
    return predicted_class_name

def extract_features(file_path, mfcc, chroma, mel):
    with soundfile.SoundFile(file_path) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def predict_emotion_audio(file_path):
    features = extract_features(file_path, mfcc=True, chroma=True, mel=True)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model_audio(features_tensor)
        predicted_class = torch.argmax(output).item()
    return class_names_audio[predicted_class]

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            predicted_class_video = classify_emotion_video(frame)
            
            # Capture audio
            audio_file_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            duration = 0.005 # 3 seconds
            recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
            sd.wait()
            soundfile.write(audio_file_path, recording, samplerate=44100)

            predicted_class_audio = predict_emotion_audio(audio_file_path)

            cv2.putText(frame, f'Video Emotion: {predicted_class_video}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Audio Emotion: {predicted_class_audio}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
