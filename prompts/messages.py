from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=.5)

message = [
    SystemMessage(content="You are a helpful assistant that helps developers to write their code."),
    HumanMessage(content="Write this code in python: 'sum of first 10 natural numbers'")
]

result = model.invoke(message)

message.append(AIMessage(content=result.content))

print(message)