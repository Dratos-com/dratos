from google import genai
from google.genai import types
import base64
import os
import json
from google.oauth2 import service_account
credentials_path = os.path.join(os.getcwd(), "credentials.json")

def generate():
    credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = genai.Client(
        credentials=credentials,
        vertexai=True,
        project="pdf-server-451815",
        location="us-central1",
    )

    msg2_text1 = types.Part.from_text(text="""Cymbal Direct is an online direct-to-consumer footwear and apparel retailer headquartered in Chicago.""")
    msg4_text1 = types.Part.from_text(text="""Founded in 2008, Cymbal Direct (originally ‘Antern’) is a fair trade and B Corp certified sustainability-focused company that works with cotton farmers to reinvest in their communities.""")
    si_text1 = """Cymbal Direct is an online direct-to-consumer footwear and apparel retailer headquartered in Chicago. 

    Founded in 2008, Cymbal Direct (originally 'Antern' is a fair trade and B Corp certified sustainability-focused company that works with cotton farmers to reinvest in their communities. The price range for Cymbal clothes is typically between $50 and $300.

    In 2010, as Cymbal Group began focusing on digitally-savvy businesses that appealed to a younger demographic of shoppers, the holding company acquired Antern and renamed it Cymbal Direct. In 2019, Cymbal Direct reported an annual revenue of $7 million and employed a total of 32 employees. 

    Cymbal Direct is a digitally native retailer. 

    You are a personalized wiki of the company Cymbal."""

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="""What does Cymbal sell?""")
        ]
        ),
        types.Content(
        role="model",
        parts=[
            msg2_text1
        ]
        ),
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="""When was the company founded?""")
        ]
        ),
        types.Content(
        role="model",
        parts=[
            msg4_text1
        ]
        ),
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="""How much is the price of Cymbal clothes?""")
        ]
        ),
        types.Content(
        role="model",
        parts=[
            types.Part.from_text(text="""The price range for Cymbal clothes is typically between $50 and $300.""")
        ]
        ),
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="""Hello""")
        ]
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature = 0.2,
        top_p = 0.8,
        seed = 0,
        max_output_tokens = 1024,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        print(chunk.text, end="")

generate()