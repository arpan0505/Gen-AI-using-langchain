from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=.5)

class review (TypedDict):
    summery: str
    rating: int
    pros: str
    cons: str

structured_model = model.with_structured_output(review)

result = structured_model.invoke("I recently bought the NoiseFit Halo smartwatch and overall, I’d give it a solid 4 out of 5 stars. The design feels premium and lightweight, and the AMOLED display is bright and crisp, making it easy to read even outdoors, which I really love. Battery life is impressive too—I can go almost a week on a single charge, and the fitness tracking features like heart rate monitoring and sleep analysis are fairly accurate, giving me good insights into my daily routine. On the downside, the companion app still feels a bit clunky and sometimes struggles to sync properly, plus I noticed that notifications can be slightly delayed at times. Still, for the price, it’s a great buy and I’m happy with how it balances style and functionality.")

print(result) 
