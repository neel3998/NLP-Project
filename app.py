import streamlit as st
import base64
from transformers import pipeline
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
# set_png_as_page_bg('E://Sem 7//NLP//Webapp//Background_image.jpg')

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
    model=pipeline('question-answering')
    return model
    
qa=load_model()
st.title("Try out our application!!")
articles=st.text_area('Please enter your article')
quest =st.text_input('Ask your question based on the article')
model_name=st.radio('Select Model',['xlm-roberta', 'mbert', 'bert'])
Lang=st.radio('Select Answer Text Language',['Hindi','English','Tamil'])
button=st.button('Answer')

with st.spinner('Finding Answer...'):
    if button and articles:
        answers=qa(question=quest, context=articles)
        
        if Lang=='Hindi':
            desty1='hi'
        elif Lang=='Tamil':
            desty1='ta'
        else:
            desty1='en'
        
        st.success(answers['answer'])
