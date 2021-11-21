import streamlit as st
import base64
import time
import streamlit.components.v1 as components
from googletrans import Translator
import transformers
translator = Translator(service_urls=['translate.googleapis.com'])
st.set_page_config(page_title='NLP Question-Answering App')
hide_menu_style=""" 
<style> 
#MainMenu {visibility:hidden;}
footer{visibility:hidden;}
</style>
"""
# st.markdown(hide_menu_style,unsafe_allow_html=True)

import base64

print(transformers.__version__)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
# set_png_as_page_bg("E://Sem 7//NLP//Webapp//NLP-Project//Background_image.jpg")

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <style>
    .jumbotron{
        background: background;  
    }
    .container{
    color: white
    }
    .display-4{
    text-align: center;
    justify-content: center;
    font-size: xx-large;
    font-weight: bold;
    }
    .lead{
   
    justify-content: center;
    align-items: center;
    text-align: center;
    font-size: x-large;
    }
  
    
    </style>
    <div class="jumbotron jumbotron-fluid">
  <div class="container">
    <h1 class="display-4">CS 613- NLP</h1>
    <h1 class="display-4">CHAII Question Answering Project</h1>
    <p class="lead">Get answers of questions based on your content.</p>
   
    <p class="lead">Prof. Mayank Singh</p>
    <p class="lead">Mentor: Harsh Patel</p>
    <p class="lead"> -By Neel Patel, Rahul Gupta, Rishi Patidar & Suryansh Kumar.</p>
  </div>
</div>

    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    
    """,
    height=400, width= 800
)

@st.cache(allow_output_mutation=True)

def load_model():
    
    # model=pipeline('question-answering')
    
    print(1)
    return 
import ast
import os
import pandas as pd
mod=load_model()
st.title("Try out our application!!")
articles=st.text_area('Please enter your article')
quest =st.text_input('Ask your question based on the article')
test = pd.DataFrame(columns = ['id','context', 'question', 'language'])
model_name=st.radio('Select Model',['xlm-roberta', 'mbert', 'muril'])
Lang=st.radio('Select Answer Text Language',['Hindi','English','Tamil'])
test = test.append({'id': '7dihav832', 'context': articles, 'question': quest,'language':Lang}, ignore_index=True)
test = test.tail(1)
test.to_csv('test.csv', index=False)
button=st.button('Answer')

#@st.cache  # 👈 Added this
def expensive_computation_xlm():
    time.sleep(2)  # This makes the function take 2s to run
    # exec(open("final_1.py",encoding='utf-8').read())
    print("Opening model file")
    os.system('python final_xlm.py')

    print(2)
    file1 = open("answers.txt", "r",encoding='utf-8')
    contents = file1.read()
    print(contents)
    return contents

#@st.cache  # 👈 Added this
def expensive_computation_muril():
    time.sleep(2)  # This makes the function take 2s to run
    # exec(open("final_1.py",encoding='utf-8').read())
    print("Opening model file")
    os.system('python final_muril.py')

    print(2)
    file1 = open("answers.txt", "r",encoding='utf-8')
    contents = file1.read()
    print(contents)
    return contents

def expensive_computation_mbert():
    time.sleep(2)  # This makes the function take 2s to run
    # exec(open("final_1.py",encoding='utf-8').read())
    print("Opening model file")
    os.system('python final_mbert.py')

    print(2)
    file1 = open("answers.txt", "r",encoding='utf-8')
    contents = file1.read()
    print(contents)
    return contents


with st.spinner('Finding Answer...'):
    if button and articles:
        if model_name == 'xlm-roberta':
            a = expensive_computation_xlm()
        elif model_name == 'muril':
            a = expensive_computation_muril()
        elif model_name == 'mbert':
            a = expensive_computation_mbert()
        answers = ast.literal_eval(a)
        
        if Lang=='Hindi':
            desty1='hi'
        elif Lang=='Tamil':
            desty1='ta'
        else:
            desty1='en'
        st.success(answers[0])

# translator.translate(str(answers[0]) ,dest=desty1).text
