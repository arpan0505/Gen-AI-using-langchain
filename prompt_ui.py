from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research assisant')
user_input = st.text_input('Enter your prompt here:')
if st.button('Generate'):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=.5)
    result = model.invoke(user_input)
    st.write(result.content)