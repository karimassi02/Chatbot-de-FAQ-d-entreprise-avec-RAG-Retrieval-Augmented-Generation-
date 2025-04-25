import streamlit as st
from openai import OpenAI



st.title("Chatbot de FAQ d'entreprise avec RAG (Retrieval Augmented Generation)")       
client = OpenAI(
    api_key="REMOVED"
)
audio_value = st.audio_input("Record a voice message")

if audio_value:
    st.audio(audio_value)
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_value
    #file= open("multilingual.wav", "rb")
    )
    st.write(transcription.text)