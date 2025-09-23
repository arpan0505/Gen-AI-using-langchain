from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

schema = [
    ResponseSchema(name='Fact 1', description='The first fact about the topic'),
    ResponseSchema(name='Fact 2', description='The second fact about the topic'),
    ResponseSchema(name='Fact 3', description='The third fact about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide three interesting facts about the following topic: {topic}. \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "Zubeen Garg"})
print(result)