from langchain_openai import ChatOpenAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

class Feedback(BaseModel):
    category: Literal["positive", "negative"] = Field(description="Give the sentiment category of the feedback")
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)    

prompt1 = PromptTemplate(
    template="Classify the following feedback into one of these categories: positive or negative\n\nText: {feedback}\n {format_instructions}", 
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classify_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="The following customer feedback is positive. Write a warm, appreciative, and professional response thanking the customer:\n\n{feedback}", 
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="The following customer feedback is negative. Write a polite and professional company response that acknowledges the complaint, apologizes for the issue, and offers assistance:\n\n{feedback}", 
    input_variables=["feedback"]
)


branch_chain = RunnableBranch(
    (lambda x: x.category == "positive", prompt2 | model | parser),
    (lambda x: x.category == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: 'No proper sentiment detected.')
)

final_chain = classify_chain | branch_chain

print(final_chain.invoke({'feedback': "I absolutely love this product! The design is sleek, the performance is excellent, and it has made my daily tasks so much easier. Definitely worth every penny."}))
