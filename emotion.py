import streamlit as st
import os
from transformers import pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
candidate_labels = ["happy", "sad", "angry","anxious","bored", "horrified", "confused", "excited"]

st.title("Emotion App")
input = st.text_input("Write what you are thinking:")

if len(input) > 0:
	classifier = pipeline("zero-shot-classification", device=0)
	x = classifier(input,candidate_labels)
	st.write(x)
else:
	pass
