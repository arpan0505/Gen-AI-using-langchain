from langchain_openai import ChatOpenAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Models
model1 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# Prompts
prompt1 = PromptTemplate(
    template="Generate a short and simple notes from the following article:\n\n{article}", 
    input_variables=["article"]
)
prompt2 = PromptTemplate(
    template="Generate 5 quizzes from the following article:\n\n{article}", 
    input_variables=["article"]
)
prompt3 = PromptTemplate(
    template="Merge the following notes and quizzes into a single document:\n\nNotes:\n{notes}\n\nQuizzes:\n{quizzes}", 
    input_variables=["notes", "quizzes"]
)

parser = StrOutputParser()

# Parallel execution: run both note-generation and quiz-generation
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quizzes': prompt2 | model2 | parser
})

# Series: merge results
series_chain = prompt3 | model1 | parser

# Final chain: parallel → series
final_chain = parallel_chain | series_chain

# Input article
article = """The Rise of Artificial Intelligence in Everyday Life

Artificial Intelligence (AI) is no longer a futuristic concept confined to research labs or science fiction movies; it has seamlessly woven itself into our daily routines, often in ways we barely notice. From voice assistants like Siri and Alexa to personalized recommendations on Netflix and Amazon, AI has become a silent companion, shaping the way we work, learn, and entertain ourselves.

One of the most visible impacts of AI is in healthcare. Modern hospitals now employ AI-powered diagnostic tools capable of detecting diseases at earlier stages than traditional methods. For example, machine learning algorithms can analyze medical images with remarkable accuracy, often outperforming human specialists in detecting subtle anomalies. This not only speeds up diagnosis but also ensures more effective treatments.

The education sector is also undergoing a quiet revolution. Adaptive learning platforms tailor lessons to individual students’ needs, ensuring that no learner is left behind. Instead of a one-size-fits-all approach, AI personalizes the learning journey, helping students strengthen weak areas while progressing faster in subjects they excel at.

Yet, while the benefits of AI are vast, the technology also raises important ethical questions. Issues such as data privacy, algorithmic bias, and job displacement demand careful consideration. Policymakers, technologists, and society at large must collaborate to ensure that AI evolves in a direction that benefits all, without exacerbating inequality or infringing on human rights.

In essence, AI is a double-edged sword—its potential to improve lives is immense, but so is the responsibility that comes with it. As the technology continues to grow, the challenge lies not in resisting AI but in guiding its development wisely. The future will not be about humans versus machines, but about how humans and machines can work together to create a more efficient, equitable, and innovative world.
"""

# Run final chain
result = final_chain.invoke({'article': article})
print(result)
