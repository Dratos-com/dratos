from pydantic import BaseModel, ConfigDict
import outlines.models as models
from outlines import generate

model = models.openai("gpt-4o-mini")

class Person(BaseModel):
    model_config = ConfigDict(extra='forbid')  # required for openai
    first_name: str
    last_name: str
    age: int

generator = generate.json(model, Person)
print(generator("current indian prime minister on january 1st 2023"))
# Person(first_name='Narendra', last_name='Modi', age=72)

generator = generate.choice(model, ["Chicken", "Egg"])
print(generator("Which came first?"))
# Chicken