from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me a fictional character with the following details in JSON format: name, age, occupation, and a brief biography. \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser   
result = chain.invoke({})
print(result)