# NLP-Project

# Demo of App:
Image 1

![image](https://user-images.githubusercontent.com/74199478/142750765-d13e7e0f-dcde-4e40-a202-80b7bfe4f66d.png)

Image 2
![image](https://user-images.githubusercontent.com/74199478/142750780-72ad952f-99d0-4350-8625-fc59322e549a.png)

# Download the pre-trained model files:
Download xlm model file from [here](https://drive.google.com/file/d/1U7v7LcOUIXzyhBsoPmawqALS3AkfYUVB/view?usp=sharing) and put the 'pytorch_model.bin' file in xlm folder.

Download mbert model file from [here](https://drive.google.com/file/d/1tYeCMKztIZfjdW0l6SZlTKC0fN-VTBI7/view?usp=sharing) and put the 'pytorch_model.bin' file in mbert folder.

Download muril model file from [here](https://drive.google.com/file/d/1L0MufcY2B5f9vhYtGXxb0uXaCSI3xzif/view?usp=sharing) and put the 'pytorch_model.bin' file in muril folder.


# Requirements
streamlit(1.2.0), Command to install: pip install streamlit

transformers(4.12.4), Command to install: pip install transformers[tf-cpu]

torch, command to install: pip install torch

sklearn, command to install: pip install scikit-learn

numpy, Command to install: pip install numpy

# How to run the app
1. Clone the repository
2. Put the downloaded pre-trained models in the same folder as app.py
3. Run the following command in the terminal: "streamlit run app.py"

# How to experiment
1. Add the context and question.
2. Choose the model and language of answer text. Click on the answer button.
