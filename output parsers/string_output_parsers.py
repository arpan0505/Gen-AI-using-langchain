from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

#template 1

template1 = PromptTemplate(
    template="Write a brief description about {topic}.", 
    input_variables=["topic"]
)

#template 2
template2 = PromptTemplate(
    template="Write a 5 line summary about that \n {text}.", 
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'write a article about heartthrob of assam Zubin garg'})
print(result)
