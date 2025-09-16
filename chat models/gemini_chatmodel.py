from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=.9)
result = model.invoke("Write a brief describtion about assam")
print(result.content)