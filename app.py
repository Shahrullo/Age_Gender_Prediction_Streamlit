import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

st.title('Welcome to Project Age Gender Prediction')
st.text("")
st.text("")
st.write("""
    Upload your image. The image you select will be fed through the Deep Neural Network in real-time 
    and the output will be displayed to the screen. 
    This project is based on my research work.
    """)

try:
    file = st.file_uploader('Upload An Image')

    pretrained_size = 224

    test_transform = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    model = models.densenet161()
    device = torch.device('cpu')

    classes_age = {
        0: '0-2 years',
        1: '4-6 years',
        2: '8-13 years',
        3: '15-20 years',
        4: '25-32 years',
        5: '38-43 years',
        6: '48-53 years',
        7: '60 years and above',
    }

    classes_gender = {
        0: 'male',
        1: 'female'
    }


    def test_on_a_class(c, image):
        with torch.no_grad():
            if c == 'age':
                model.classifier = nn.Linear(in_features=2208, out_features=8)
            elif c == 'gender':
                model.classifier = nn.Linear(in_features=2208, out_features=2)
            model.to(device)
            model.load_state_dict(torch.load(f'{c}.pt'))
            model.eval()
            output = model(image)
            output = torch.max(output, 1)[1].to(device)
            if c == 'age':
                result = f'{classes_age[output.item()]}'
            elif c == 'gender':
                result = f'{classes_gender[output.item()]}'
        return result


    def test_img(img_path):
        image = Image.open(img_path)
        image = test_transform(image)
        image.unsqueeze_(0)
        image = image.to(device)
        return test_on_a_class('age', image), test_on_a_class('gender', image)

    result = test_img(file)



    img = Image.open(file)

    st.subheader('Here is the image you selected and the result')
    st.write(f'Your Age between {result[0]} old and You are {result[1]}')
#     resized = img.resize((384, 384))
    st.image(img)
    
except:
    st.error('Please upload your photo')
