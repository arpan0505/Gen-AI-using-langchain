from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=.5)

class review (TypedDict):
    key_themes: Annotated[list[str], "Key themes mentioned in the review"]
    summery: Annotated[str, "A brief summary of the review"]
    rating: Annotated[int, "Rating out of 5"]
    sentiment: Annotated[str, "Overall sentiment of the review"]
    pros: Annotated[Optional[list[str]], "Positive aspects mentioned in the review"]
    cons: Annotated[Optional[list[str]], "Negative aspects mentioned in the review"]

structured_model = model.with_structured_output(review)

result = structured_model.invoke("""I recently purchased these premium over-ear headphones after extensive research and testing of various models, and I must say the experience has been a blend of delight and minor frustration; the sound quality is nothing short of phenomenal, with crisp highs, rich mids, and deep, resonant bass that makes listening to music a truly immersive experience, and the active noise-cancellation works remarkably well even in busy environments, allowing me to focus without distractions, yet I noticed that under heavy use, the battery drains slightly faster than advertised, which can be inconvenient during long trips, while the build quality feels solid and comfortable, with ear cups that remain snug without causing fatigue, the companion app, though feature-rich, occasionally lags and can be unintuitive at times, leaving me wishing for smoother software optimization, nevertheless, the intuitive touch controls, quick pairing with multiple devices, and thoughtful design choices make these headphones a joy to use daily, combining both style and functionality, so overall, while not absolutely perfect, they deliver a near-exceptional listening experience that makes them well worth the investment.""")

print(result) 
