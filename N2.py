# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 02:11:41 2023

@author: Shashank
"""

import streamlit as st
import pickle

# Define global paths to pickle files
MODEL_FILE_PATH = 'model.pkl'
VECTORIZER_FILE_PATH = 'vectorizer.pkl'

# Load the pickled model and TfidfVectorizer
with open(MODEL_FILE_PATH, 'rb') as file:
    model = pickle.load(file)

with open(VECTORIZER_FILE_PATH, 'rb') as file:
    vectorizer = pickle.load(file)


st.title("Multi-Label Comment Classification")

comment = st.text_area('Enter a comment')

if st.button('Classify'):
    if comment:
        # Preprocess and transform the input
        processed_comment = vectorizer.transform([comment])

        # Make predictions
        predictions = model.predict(processed_comment)

        # Display the predictions
        st.write('The comment labels are:')
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'clean']
        for idx, label in enumerate(labels):
            st.write(f"{label}: {predictions[0][idx]}")
    else:
        st.warning('Please enter a comment for classification.')
