import streamlit as st
# import tkinter as tk
# from tkinter import messagebox
import speech_recognition as sr
import re
from pydub import AudioSegment
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import difflib

def Similarity_check(text1, text2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    similarity_score = np.dot(embeddings[0].detach().numpy(), embeddings[1].detach().numpy()) / (
        np.linalg.norm(embeddings[0].detach().numpy()) * np.linalg.norm(embeddings[1].detach().numpy())
    )
    return similarity_score

def missing_words(text1,text2):
    words1 = text1.split()
    words2 = text2.split()
    differ = difflib.Differ()
    diff = list(differ.compare(words1, words2))
    missing_words = [word[2:] for word in diff if word.startswith('- ')]
    # print("Missing words from text1 to text2:")
    # print(" ".join(missing_words))
    return " ".join(missing_words)

def extra_words(text1,text2):
    words1 = text1.split()
    words2 = text2.split()
    differ = difflib.Differ()
    diff = list(differ.compare(words1, words2))
    extra_words = [word[2:] for word in diff if word.startswith('+ ')]
    # print("Extra words in text2 compared to text1:")
    # print(" ".join(extra_words))
    return " ".join(extra_words)

# def show_popup(score):
#     root = tk.Tk()
#     root.withdraw()  # Hide the main tkinter window
#     if score>=0.7:
#         messagebox.showinfo("Congratulations!🥳", "🌟🌟🌟🌟🌟\nWell done kid! You completed this level 🎉")
#     elif score<0.7 and score>=0.5:
#         messagebox.showinfo("Congratulations!🥳", "⭐⭐⭐⭐Well done kid..........!\t You completed this level 🎉")
#     elif score<0.5 and score>=0.3:
#         messagebox.showinfo("Congratulations!🥳", "⭐⭐Well done kid..........!\t You completed this level 🎉")
#     else:
#         messagebox.showinfo("Sorry kidd!", " You failed this level 🎉")

st.title("Speech Enhancement")
st.write("Upload a text file and display its contents.")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
st.write("Upload an audio file and convert it to text.")
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    r = sr.Recognizer()
    with st.spinner("Converting audio to text..."):
        try:
            audio_data = AudioSegment.from_file(audio_file)
            audio_data = audio_data.set_channels(1)
            audio_data.export("temp.wav", format="wav")
            with sr.AudioFile("temp.wav") as source:
                audio_text = r.record(source)
            text = r.recognize_google(audio_text)
            st.success("Conversion complete!")
            st.subheader("Converted Text:")
            st.write(text)
            st.success("Similarity....")

            # data = np.loadtxt("/home/jayaprakash/machine learning/stt/speech.txt",dtype='str')
            # string=" ".join(data)
            # string1=re.sub(r'[^\w\s]','',string)
            # print(string1)

            string1 = None
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                string1 = file_bytes.decode("utf-8")
                st.subheader("Uploaded Text:")
                st.write(string1)
            else:
                st.write("Please upload a text file.")

            string2=text
            st.write(Similarity_check(string1,string2))
            # show_popup(Similarity_check(string1,string2))

            st.subheader("Missing words:")
            st.write(missing_words(string1,string2))
            st.subheader("Extra words:")
            st.write(extra_words(string1,string2))

        except Exception as e:
            st.error(f"Error: {e}")

