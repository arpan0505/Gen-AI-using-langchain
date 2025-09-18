from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=.5)
result = model.invoke("Write a brief describtion about assam")
print(result.content)