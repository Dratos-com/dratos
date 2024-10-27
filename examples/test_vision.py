"""
* WORKING

What this script does:
Generate a tool request using an LLM.

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY
"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAI


def vision(prompt: str):
    """
    Generate a text completion for a given an image.

    >>> vision("What's the meme about? (very short answer)")
    'The meme humorously contrasts a minor issue (a spider) with a dramatic or chaotic background (a house on fire), suggesting that the speaker is downplaying a serious situation.'
    """
    llm = LLM(
        model_name="gpt-4o-mini", 
        engine=OpenAI(),
    )

    agent = Agent(
        name="agent",
        llm=llm,
        # verbose=True,
    )

    return agent.sync_gen({"text": prompt, "meme.jpg": "https://ichef.bbci.co.uk/images/ic/1920xn/p072ms6r.jpg"})

# vision("What's the meme about?")

def vision2(prompt: str):
    """
    Generate a text completion for a given an image.

    >>> vision2("What are the two memes about? (very short answer)")
    '1. **Serious Cat**: A cat portraying a serious demeanor, often used to sarcastically emphasize seriousness in situations.2. **Honest Work Guy**: A man appreciating simple, honest labor, typically used to highlight humility and modesty in work.'
    """
    llm = LLM(
        model_name="gpt-4o-mini", 
        engine=OpenAI(),
    )

    agent = Agent(
        name="agent",
        llm=llm,
        # verbose=True,
    )

    return agent.sync_gen({"text": prompt,
                            "meme.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/CatLolCatExample.jpg/170px-CatLolCatExample.jpg",
                            "meme2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Farmer_meme_with_apostrophe.jpg/220px-Farmer_meme_with_apostrophe.jpg",
                            })

# vision2("What are the two memes about?")

def vision_base64(prompt: str):
    """
    Generate a text completion for a given an image.

    >>> vision_base64("What's the meme about? (very short answer)")
    'The meme features a serious-looking cat and is about authority, often used humorously to convey sternness in a playful way.'
    """
    llm = LLM(
        model_name="gpt-4o-mini", 
        engine=OpenAI(),
    )

    agent = Agent(
        name="agent",
        llm=llm,
        # verbose=True,
    )

    import base64
    from urllib.request import urlopen

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/CatLolCatExample.jpg/170px-CatLolCatExample.jpg"
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    image_data = urlopen(image_url).read()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    return agent.sync_gen({"text": prompt,
                            "meme.jpeg": base64_image
                            })

# vision_base64("What's the meme about? (very short answer)")
