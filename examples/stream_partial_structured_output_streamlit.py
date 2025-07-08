"""
* WORKING

What this script does:
Simple demo of Pydantic's experimental_allow_partial feature.
Shows detected countries as horizontal badges as they stream.

Requirements:
   - OPENAI_API_KEY in your .env file

To run: streamlit run examples/stream_partial_structured_output_streamlit.py
"""

import sys
import os
import asyncio
import warnings
from typing import List, Optional

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

from dratos import LLM, Agent, LiteLLMEngine
from pydantic import BaseModel, TypeAdapter, PydanticExperimentalWarning
import streamlit as st

warnings.filterwarnings('ignore', category=PydanticExperimentalWarning)

class CountryCapital(BaseModel):
    country: str
    capital: str
    population: Optional[int] = None
    currency: Optional[str] = None

class EuropeData(BaseModel):
    region: str = "Europe"
    countries: List[CountryCapital]
    total_countries: Optional[int] = None

st.title("🌍 Partial Streaming Demo")
st.write("Watch countries appear as badges while JSON streams!")

# Display areas
raw_container = st.container()
badges_container = st.container()

with raw_container:
    st.subheader("Raw Stream")
    raw_placeholder = st.empty()

with badges_container:
    st.subheader("Detected Countries")
    badges_placeholder = st.empty()

@st.cache_resource
def get_agent():
    llm = LLM(model_name="gpt-4o-mini", engine=LiteLLMEngine(provider="openai"))
    return Agent(name="simple_agent", llm=llm, response_model=EuropeData, verbose=False)

def create_country_badges(countries):
    """Create horizontal badges for countries"""
    if not countries:
        return "⏳ Waiting for countries..."
    
    badges_html = ""
    for country in countries:
        if hasattr(country, 'country') and country.country:
            capital = getattr(country, 'capital', '?')
            badges_html += f"""
            <span style="
                display: inline-block; 
                margin: 2px 4px; 
                padding: 4px 8px; 
                background-color: #e1f5fe; 
                border: 1px solid #0277bd; 
                border-radius: 12px; 
                font-size: 14px;
                color: #0277bd;
            ">
                🏛️ {country.country} ({capital})
            </span>
            """
    return badges_html

async def stream_and_parse():
    agent = get_agent()
    ta = TypeAdapter(EuropeData)
    
    prompt = "List 8 European countries with capitals in JSON format: {region: 'Europe', countries: [{country: 'France', capital: 'Paris'}, ...]}"
    
    accumulated_text = ""
    previous_count = 0
    
    async for chunk in agent.async_gen({"text": prompt}):
        accumulated_text += chunk
        raw_placeholder.code(accumulated_text, language="json")
        
        try:
            result = ta.validate_json(accumulated_text, experimental_allow_partial=True)
            current_count = len(result.countries) if result.countries else 0
            
            # Only update badges if we detected new countries
            if current_count > previous_count:
                badges_html = create_country_badges(result.countries)
                badges_placeholder.markdown(badges_html, unsafe_allow_html=True)
                previous_count = current_count
                
        except:
            pass  # Continue streaming
        
        await asyncio.sleep(0.1)
    
    return accumulated_text

if st.button("🚀 Start Demo", type="primary"):
    with st.spinner("Streaming..."):
        try:
            final_text = asyncio.run(stream_and_parse())
            st.success(f"✅ Complete! Streamed {len(final_text)} characters")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.button("🔄 Clear"):
    raw_placeholder.empty()
    badges_placeholder.empty()
    st.rerun() 