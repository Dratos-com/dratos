"""
* WORKING

What this script does:
Generate a tool request using an LLM.

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY


BE AWARE:
Using OpenAI built-in response_model support, only the following types are supported:
String
Number
Boolean
Object
Array
Enum
anyOf
This is in line with JSON data type support (which doesn’t have support for datetime).
"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAIEngine

from pydantic import BaseModel


def structured_generation(prompt):
    """
    Generate a text completion for a given prompt.

    >>> structured_generation("What is the capital of Canada?")
    Capital(capital='Ottawa', country='Canada')
    """
    llm = LLM(
        model_name="gpt-4o-2024-08-06", 
        engine=OpenAIEngine(),
    )

    class User(BaseModel):
        name: str

    

    class Position(BaseModel):
        description: str
        choice: str
        quantity: float
        
    class Prediction(BaseModel):
        description: str
        choice: str
    
    class MarketData(BaseModel):
        name: str
        description: str
        predictions: list[Prediction]
        date_start:str
        date_end_prediction:str
        date_outcome:str

    agent = Agent(
        name="mako setup",
        llm=llm,
        response_model=MarketData,
        response_validation=True,
        verbose=True,
    )

    return agent.sync_gen({"text": prompt})

structured_generation("""
                      You are a Makoa company assistant helping users setup prediction markets.
                      Prediction Markets on Makao are composed of Markets where you can make predictions. People may position them self on one or multiple preductions with a qunatity of USDT.
                      
                      \n\n
                      ### Market Title:
"Will Telegram Be Banned in France by 2025?"

**Market Overview:**  
Predict if Telegram will be officially banned in France by 31/12/25, based on government action that legally restricts or blocks access within French borders.

**Participation Deadline:**  
31/12/24 23:59

**Decision Deadline:**  
07/01/26 23:59

**Predictions:**  
1. **Telegram is banned**  
   **Criteria:** Valid if an official government order or legal enforcement action blocks access to Telegram within France by 31/12/25.

2. **Telegram is not banned**  
   **Criteria:** Valid if Telegram remains accessible in France without any government ban by 31/12/25.

**Notes:**  
- Initial bets: Nico placed $1 on "Telegram is banned" and $10 on "Telegram is not banned."
- Partial restrictions that do not block access do not qualify as a ban.
                      """)

