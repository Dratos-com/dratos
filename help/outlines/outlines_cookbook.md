Classification
Classification is a classic problem in NLP and finds many applications: spam detection, sentiment analysis, triaging of incoming requests, etc. We will use the example of a company that wants to sort support requests between those that require immediate attention (URGENT), those that can wait a little (STANDARD). You could easily extend the example by adding new labels.

This tutorial shows how one can implement multi-label classification using Outlines. We will use two functionalities of the library: generate.choice and generate.json.

As always, we start with initializing the model. Since we are GPU poor we will be using a quantized version of Mistal-7B-v0.1:


import outlines

model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")
We will use the following prompt template:


@outlines.prompt
def customer_support(request):
    """You are an experienced customer success manager.

    Given a request from a client, you need to determine when the
    request is urgent using the label "URGENT" or when it can wait
    a little with the label "STANDARD".

    # Examples

    Request: "How are you?"
    Label: STANDARD

    Request: "I need this fixed immediately!"
    Label: URGENT

    # TASK

    Request: {{ request }}
    Label: """
Choosing between multiple choices
Outlines provides a shortcut to do multi-label classification, using the outlines.generate.choice function to initialize a generator. Outlines uses multinomial sampling by default, here we will use the greedy sampler to get the label with the highest probability:


from outlines.samplers import greedy

generator = outlines.generate.choice(model, ["URGENT", "STANDARD"], sampler=greedy())
Outlines supports batched requests, so we will pass two requests to the model:

requests = [
    "My hair is one fire! Please help me!!!",
    "Just wanted to say hi"
]

prompts = [customer_support(request) for request in requests]
We can now asks the model to classify the requests:


labels = generator(prompts)
print(labels)
# ['URGENT', 'STANDARD']
Now, you might be in a hurry and don't want to wait until the model finishes completion. After all, you only need to see the first letter of the response to know whether the request is urgent or standard. You can instead stream the response:


tokens = generator.stream(prompts)
labels = ["URGENT" if "U" in token else "STANDARD" for token in next(tokens)]
print(labels)
# ['URGENT', 'STANDARD']
Using JSON-structured generation
Another (convoluted) way to do multi-label classification is to JSON-structured generation in Outlines. We first need to define our Pydantic schema that contains the labels:


from enum import Enum
from pydantic import BaseModel


class Label(str, Enum):
    urgent = "URGENT"
    standard = "STANDARD"


class Classification(BaseModel):
    label: Label
and we can use generate.json by passing this Pydantic model we just defined, and call the generator:


generator = outlines.generate.json(model, Classification, sampler=greedy())
labels = generator(prompts)
print(labels)
# [Classification(label=<Label.urgent: 'URGENT'>), Classification(label=<Label.standard: 'STANDARD'>)]


Named entity extraction
Named Entity Extraction is a fundamental problem in NLP. It involves identifying and categorizing named entities within a document: people, organization, dates, places, etc. It is usually the first step in a more complex NLP worklow. Here we will use the example of a pizza restaurant that receives orders via their website and need to identify the number and types of pizzas that are being ordered.

Getting LLMs to output the extracted entities in a structured format can be challenging. In this tutorial we will see how we can use Outlines' JSON-structured generation to extract entities from a document and return them in a valid JSON data structure 100% of the time.

As always, we start with initializing the model. We will be using a quantized version of Mistal-7B-v0.1 (we're GPU poor):


import outlines

model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")
And we will be using the following prompt template:


@outlines.prompt
def take_order(order):
    """You are the owner of a pizza parlor. Customers \
    send you orders from which you need to extract:

    1. The pizza that is ordered
    2. The number of pizzas

    # EXAMPLE

    ORDER: I would like one Margherita pizza
    RESULT: {"pizza": "Margherita", "number": 1}

    # OUTPUT INSTRUCTIONS

    Answer in valid JSON. Here are the different objects relevant for the output:

    Order:
        pizza (str): name of the pizza
        number (int): number of pizzas

    Return a valid JSON of type "Order"

    # OUTPUT

    ORDER: {{ order }}
    RESULT: """
We now define our data model using Pydantic:


from enum import Enum
from pydantic import BaseModel

class Pizza(str, Enum):
    margherita = "Margherita"
    pepperonni = "Pepperoni"
    calzone = "Calzone"

class Order(BaseModel):
    pizza: Pizza
    number: int
We can now define our generator and call it on several incoming orders:


orders = [
    "Hi! I would like to order two pepperonni pizzas and would like them in 30mins.",
    "Is it possible to get 12 margheritas?"
]
prompts = [take_order(order) for order in orders]

generator = outlines.generate.json(model, Order)

results = generator(prompts)
print(results)
# [Order(pizza=<Pizza.pepperonni: 'Pepperoni'>, number=2),
#  Order(pizza=<Pizza.margherita: 'Margherita'>, number=12)]
There are several ways you could improve this example:

Clients may order several types of pizzas.
Clients may order drinks as well.
If the pizza place has a delivery service we need to extract the client's address and phone number
Clients may specify the time for which they want the pizza. We could then check against a queuing system and reply to them with the estimated delivery time.
How would you change the Pydantic model to account for these use cases?

Generate a synthetic dating profile from a description
In this example we will see how we can use Outlines to generate synthetic data for a dating application. This example was originally contributed by Vibhor Kumar.


from dataclasses import dataclass
from enum import Enum

import torch
import transformers
from pydantic import BaseModel, conlist, constr

import outlines
Defining the profile with Pydantic
Here a dating profile will consist in a biography, a job, a list of interests and two question-answer pairs. The questions are written in advance by the team, and the users are asked to provide an answer:


class QuestionChoice(str, Enum):
    A = "The key to my heart is"
    B = "The first item on my bucket list is"
    C = "Perks of dating me"
    D = "Message me if you also love"
    E = "People would describe me as"
    F = "I can beat you in a game of"

@dataclass
class QuestionAnswer:
    question: QuestionChoice
    answer: str
Users need to provide a short biography, with a minimum of 10 and a maximum of 300 characters. The application also limits job descriptions to 50 characters. In addition to the question-answer pairs, the user is required to provide a list of between 1 and 5 interests:


class DatingProfile(BaseModel):
    bio: constr(str, min_length=10, max_length=300)
    job: constr(str, max_lengt=50)
    interests: conlist(str, min_length=1, max_length=5)  # type: ignore
    qna1: QuestionAnswer
    qna2: QuestionAnswer
Prompt template and examples
We will ask the model to generate profiles from a high-level description:


@dataclass
class Example:
    description: str
    profile: DatingProfile
We will use Outlines' prompt templating abilities to generate the prompt for us. This help clearly separate the general prompting logic from what is specific to an example.


@outlines.prompt
def dating_profile_prompt(description: str, examples: list[Example]):
    """
    You are a world-renowned matchmaker who understands the modern dating
    market. Your job is to generate dating app profiles for male clients
    interested in women based on a provided description. The profiles should be
    authentic, show off their strengths, and maximize their likelihood of
    getting matches on dating apps.  Here are some examples of past clients that
    you have successfully created profiles for:

    {% for example in examples %}
    Description:
    {{ example.description }}
    Profile:
    {{ example.profile }}
    {% endfor %}

    Here is the new client who you need to create a profile for:
    Description: {{ description }}
    Profile:
    """
We will provide the model with several few-shot examples:


samples: list[Example] = [
    Example(
        description="I'm an author and former professional soccer player living in Seattle who publishes popular fiction books. A typical day for me starts by hanging out with my cat, drinking a coffee, and reading as much as I can in a few hours. Then, I'll prepare a quick smoothie before starting to write for a few hours, take a break with soccer or running a few miles, and finally meet friends for dinner at a new, hip restaurant in the evening. Sometimes we go axe-throwing afterwards, or play poker, or watch a comedy show, or visit a dive bar. On my vacations, I travel extensively to countries South America, Europe, and Asia, with the goal of visiting them all!",
        profile=DatingProfile(
            bio="Adventurer, dreamer, author, and soccer enthusiast. Life’s too short to waste time so I make the most of each day by exploring new places and playing with my friends on the pitch. What’s your favorite way to get out and have fun?",
            job="Famous Soccer Player -> Famous Author",
            interests=["Soccer", "Travel", "Friends", "Books", "Fluffy Animals"],
            qna1=QuestionAnswer(
                question=QuestionChoice.B, answer="swim in all seven oceans!"
            ),
            qna2=QuestionAnswer(
                question=QuestionChoice.E,
                answer="fun-loving, adventurous, and a little bit crazy",
            ),
        ),
    ),
    Example(
        description="I run my company and build houses for a living. I'm a big fan of the outdoors and love to go hiking, camping, and fishing. I don't like video games, but do like to watch movies. My love language is home-cooked food, and I'm looking for someone who isn't afraid to get their hands dirty.",
        profile=DatingProfile(
            bio="If you're looking for a Montana man who loves to get outdoors and hunt, and who's in-tune with his masculinity then I'm your guy!",
            job="House Construction Manager / Entrepreneur",
            interests=["Hunting", "Hiking", "The outdoors", "Home-cooked food"],
            qna1=QuestionAnswer(question=QuestionChoice.A, answer="food made at home"),
            qna2=QuestionAnswer(
                question=QuestionChoice.C,
                answer="having a man in your life who can fix anything",
            ),
        ),
    ),
    Example(
        description="I run my own Youtube channel with 10M subscribers. I love working with kids, and my audience skews pretty young too. In my free time, I play Fortnite and Roblox. I'm looking for someone who is also a gamer and likes to have fun. I'm learning Japanese in my free time as well as how to cook.",
        profile=DatingProfile(
            bio="Easy on the eyes (find me on Youtube!) and great with kids. What more do you need?",
            job="Youtuber 10M+ subscribers",
            interests=["Kids", "Gaming", "Japanese"],
            qna1=QuestionAnswer(question=QuestionChoice.D, answer="anime and gaming!"),
            qna2=QuestionAnswer(question=QuestionChoice.F, answer="Fortnite, gg ez"),
        ),
    ),
]
Load the model
We will use Mosaic's MPT-7B model (requires 13GB of GPU memory) which can fit on a single GPU with a reasonable context window. We initialize it with Outlines:


config = transformers.AutoConfig.from_pretrained(
    "mosaicml/mpt-7b-8k-instruct", trust_remote_code=True
)
config.init_device = "meta"
model = outlines.models.transformers(
    model_name="mosaicml/mpt-7b-8k-instruct",
    device="cuda",
    model_kwargs={
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    },
)
JSON-structured generation of profiles
We will now generate a dating profile from a textual description of oneself:


new_description = """I'm a laid-back lawyer who spends a lot of his free-time
gaming. I work in a corporate office, but ended up here after the start-up  I
cofounded got acquired, so still play ping pong with my cool coworkers every
day.  I have a bar at home where I make cocktails, which is great for
entertaining  friends. I secretly like to wear suits and get a new one tailored
every few  months. I also like weddings because I get to wear those suits, and
it's  a good excuse for a date. I watch the latest series because I'm paying,
with my hard-earned money, for every streaming service."""

prompt = dating_profile_prompt(new_description, samples)
profile = outlines.generate.json(model, DatingProfile)(prompt)
parsed_profile = DatingProfile.model_validate_json(profile)
Results
Here are a couple of results:


{
    "bio": """I'm an ambitious lawyer with a casual and fashionable style. I love
    games and sports, but my true passion is preparing refreshing cocktails at
    home and dressing to the nines at weddings. I'm currently looking for a woman
    to show a good time to and get a kiss on the opulent suit I just had made.
    Send resume to this inbox.""",
    "job": "Lawyer",
    "interests":
    [
        "Stylish guys",
        "Gaming",
        "Ping pong",
        "Cocktails",
        "Weddings"
    ],
    "qna1":
    {
        "question": "The first item on my bucket list is",
        "answer": "be married and have a family."
    },
    "qna2":
    {
        "question": "People would describe me as",
        "answer": "charming, stylish, and funny."
    }
}

{
    "bio": """I’m a sexy lawyer with time on my hands. I love to game and
    play ping pong, but the real reason you should swipe to the right
    is because I look great in a suit. Who doesn’t love a man in a
    suit? Just saying. Send me a message if you think it’s time to take
    your dating life to the next level.""",
    "job": "Lawyer",
    "interests":
    [
        "Gaming",
        "Ping Pong",
        "Tailored Suits",
        "Weddings",
        "Streaming Services"
    ],
    "qna1":
    {
        "question": "The first item on my bucket list is",
        "answer": "simulate space but stay alive for as long as possible"
    },
    "qna2":
    {
        "question": "People would describe me as",
        "answer": "easy-going, a little nerdy but with a mature essence"
    }
}
Summarize documents using Chain of Density prompting
A good summary should be informative, concise and clear. While large language models are generally good at summarizing documents, their summaries tend to be long and contain redundant information; their information density tends to be on the lower end. This is where chain of Density, a new prompting technique, comes in. In this example we will show how one can implement chain of density with a few lines of code using Outlines, leveraging both Outline's prompt templating and its structured generation capabilities.

The article we will try to summarize is the first three paragraphs of the Alan Turing page on Wikipedia:


article = """
Alan Mathison Turing OBE FRS (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.[5] Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer.[6][7][8] He is widely considered to be the father of theoretical computer science and artificial intelligence.[9]

Born in Maida Vale, London, Turing was raised in southern England. He graduated at King's College, Cambridge, with a degree in mathematics. Whilst he was a fellow at Cambridge, he published a proof demonstrating that some purely mathematical yes–no questions can never be answered by computation. He defined a Turing machine and proved that the halting problem for Turing machines is undecidable. In 1938, he obtained his PhD from the Department of Mathematics at Princeton University. During the Second World War, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence. For a time he led Hut 8, the section that was responsible for German naval cryptanalysis. Here, he devised a number of techniques for speeding the breaking of German ciphers, including improvements to the pre-war Polish bomba method, an electromechanical machine that could find settings for the Enigma machine. Turing played a crucial role in cracking intercepted coded messages that enabled the Allies to defeat the Axis powers in many crucial engagements, including the Battle of the Atlantic.[10][11]

After the war, Turing worked at the National Physical Laboratory, where he designed the Automatic Computing Engine, one of the first designs for a stored-program computer. In 1948, Turing joined Max Newman's Computing Machine Laboratory at the Victoria University of Manchester, where he helped develop the Manchester computers[12] and became interested in mathematical biology. He wrote a paper on the chemical basis of morphogenesis[1] and predicted oscillating chemical reactions such as the Belousov–Zhabotinsky reaction, first observed in the 1960s. Despite these accomplishments, Turing was never fully recognised in Britain during his lifetime because much of his work was covered by the Official Secrets Act.[13]
"""
How Chain Of Density works
Chain Of Density starts with asking the model to generate a first long and non-specific summary. Then it asks the model to generate 4 extra summaries by proceeding in the following way:

Identify 1-3 entities missing in the previous summary;
Add all entities marked as missing in the previous step, while not dropping entities;
Make the summary more concise;
The prompt also asks the model to return a list of JSON objects that contain the missing entities and the new summary. This is where structured generation will come in handy :) The paper provides the prompt and an example:

Figure 2 in the paper

We can now implement the prompt provided in the paper:


import outlines

@outlines.prompt
def chain_of_density(article):
    """Article: {{ article }}

    You will generate increasingly concise, entity-dense summaries of the above Article.

    Repeat the following 2 steps 5 times.

    Step 1. Identify 1-3 informative Entities ("; " delimited) from the Article which are missing from the previously generated summary.
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

    A Missing Entity is:
    - Relevant: to the main story.
    - Specific: descriptive yet concise (5 words or fewer).
    - Novel: not in the previous summary.
    - Faithful: present in the Article.
    - Anywhere: located anywhere in the Article.

    Guidelines:
    - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
    - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
    - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

    Remember, use the exact same number of words for each summary.

    Answer in JSON. The JSON should be a a dictionary with key "summaries" that contains a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
    """
Note
Outlines implementation
We will use Outline's JSON-structured generation to ensure that the model's output is consistent with the format specified in the prompt. We start with defining the JSON objects that the model is asked to return using Pydantic. One JSON object that contains a list of Summary objects that contain the missing entities and new summary:


from pydantic import BaseModel, conlist

class Summary(BaseModel):
    missing_entities: str
    denser_summary: str

class Summaries(BaseModel):
    summaries: conlist(Summary, max_length=5, min_length=5)
We now generate the prompt by passing the article we want to summarize to the template. We load a quantized version of Mistral-7B using the AutoAWQ library, and then use JSON-structured generation to generate the summaries:


model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ")

prompt = chain_of_density(article)
result = outlines.generate.json(model, Summaries)(prompt)
We can now check the results:


print(result.model_dump())
# {'summaries': [
#     {
#       'missing_entities': 'English mathematician, cryptanalyst, philosopher',
#       'denser_summary': 'Alan Mathison Turing was an English mathematician, cryptanalyst, philosopher.'
#     },
#     {
#       'missing_entities': '',
#       'denser_summary': "Alan Mathison Turing was an English mathematician who was a crucial figure in WW2's Bletchley Park codebreaking centre and designed one of the first computers."
#     },
#     {
#       'missing_entities': 'cryptanalyst, studied, biology, father',
#       'denser_summary': 'Alan Mathison Turing was an English cryptanalyst, studied theoretical computer science, and contributed to mathematical biology.'
#     },
#     {
#       'missing_entities': 'biology, morphogenesis, chemical',
#       'denser_summary': 'Alan Mathison Turing was an English cryptanalyst, studied theoretical computer science, and predicted chemical reactions in morphogenesis.
#     '},
#     {
#       'missing_entities': '',
#       'denser_summary': 'Alan Mathison Turing was an English cryptanalyst, developed computer science, and made strides in mathematical biology research.'
#       }
# ]}
Not bad, considering we used a smallish model to generate the summary! Chain of Density seems to be a very effective prompting technique to generate dense summaries, even with small quantized models. Its implementation in Outlines is also very short.

Note that this is the first article I tried and it worked out of the box. Try it out on other articles, and please share the results on Twitter, or by opening a new discussion on the Outlines repository!

Large language models playing chess
In this example we will make a Phi-2 model play chess against itself. On its own the model easily generates invalid moves, so we will give it a little help. At each step we will generate a regex that only matches valid move, and use it to help the model only generating valid moves.

The chessboard
The game will be played on a standard checkboard. We will use the chess library to track the opponents' moves, and check that the moves are valid.


%pip install outlines -q
%pip install chess -q
%pip install transformers accelerate einops -q

import chess

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
The opponents
Phi-2 will be playing against itself:


from outlines import models

model = models.transformers("microsoft/Phi-3-mini-4k-instruct")
A little help for the language model
To make sure Phi-2 generates valid chess moves we will use Outline's regex-structured generation. We define a function that takes the current state of the board and returns a regex that matches all possible legal moves:


import re

def legal_moves_regex(board):
    """Build a regex that only matches valid moves."""
    legal_moves = list(board.legal_moves)
    legal_modes_str = [board.san(move) for move in legal_moves]
    legal_modes_str = [re.sub(r"[+#]", "", move) for move in legal_modes_str]
    regex_pattern = "|".join(re.escape(move) for move in legal_modes_str)
    regex_pattern = f"{regex_pattern}"
    return regex_pattern
Prompting the language model
The prompt corresponds to the current state of the board, so we start with:


prompt = "Let's play Chess. Moves: "
We update the prompt at each step so it reflects the state of the board after the previous move.

Let's play

from outlines import generate

board_state = " "
turn_number = 0
while not board.is_game_over():
    regex_pattern = legal_moves_regex(board)
    structured = generate.regex(model, regex_pattern)(prompt + board_state)
    move = board.parse_san(structured)

    if turn_number % 2 == 0 :  # It's White's turn
        board_state += board.san(move) + " "
    else:
        board_state += board.san(move) + " " + str(turn_number) + "."

    turn_number += 1

    board.push(move)

    print(board_state)
Interestingly enough, Phi-2 hates capturing.


 e4 e5 1.Nf3 Ne7 3.b4 Nf5 5.Nc3 Ne7 7.Bb5 a6 9.Na4 b6 11.c3 Nec6 13.c4 a5 15.d4 Qg5 17.Nd2 Bb7 19.d

 Build perspective-taking agents with SimToM
Prompting strategies like Chain-of-Thought (CoT) can improve LLMs' reasoning capabilities. However, they underwhelm in tasks that require keeping track of inconsistent world states. SimToM proposes a simple, two-stage prompting framework for LLMs inspired by Simulation Theory. The authors showed that this approach outperforms zero-shot prompting and CoT on ToMI and BigToM, two benchmarks with Theory of Mind questions.

In this example, we will implement SimToM with a few lines of code using Outlines' prompt templating and structured generation capabilities.

How SimToM works
SimToM calls an LLM with two consecutive prompts:

Perspective-taking: The first prompt receives a story and a character. The goal is to understand the situation based on the character's point of view and filter out the rest of the story.
Question-Answering: The second prompt receives the character's point of view from the previous step and tasks the LLM to answer a question using that context.
Figure 2 in the paper

Outlines implementation
To implement SimToM with Outlines, we will need to:

Write the prompts with prompt functions.
Define the JSON object each prompt will return using Pydantic.
Generate responses with a Mistral model using the transformers integration.
Let's dive into it!

Using Prompt Functions
With Outlines, you can write your prompts as Python functions by adding the @outlines.prompt decorator. The prompt template is contained in their docstring, and their arguments correspond to variables used in the prompt.

The authors have shared their code, prompts and data in this GitHub repository. Below, we define in Outlines the prompts they used for the ToMI dataset:


import outlines


@outlines.prompt
def perspective_taking(story: str, character: str) -> None:
    """<s>[INST] The following is a sequence of events about some characters, that takes place in multiple locations.
    Your job is to output only the events that the specified character, {{character}}, knows about.

    Here are a few rules:
    1. A character knows about all events that they do.
    2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
    3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

    Story: {{story}}
    What events does {{character}} know about? Only output the events according to the above rules, do not provide an explanation. [/INST]""" # noqa

@outlines.prompt
def simulation(events: list, name: str, question: str) -> None:
    """<s>[INST] {% for event in events %}
    {{event}}
    {% endfor %}
    You are {{name}}.
    Based on the above information, answer the following question:
    {{question}}
    You must choose one of the above choices, do not say there is not enough information. Answer with a single word, do not output anything else. [/INST]""" # noqa
JSON Structured Generation
Outlines guarantees that the LLM will return a valid JSON object, which we can specify as a Pydantic model.

We will need two Pydantic models for SimToM, one for each prompt:


from pydantic import BaseModel, Field
from typing import List


class PerspectiveTaking(BaseModel):
    """This is for the first prompt."""
    character: str = Field(description="The character we extract the events for.")
    events: List[str] = Field(description="All events that the character knows about.")


class Simulation(BaseModel):
    """This is for the second prompt."""
    answer: str
Calling an LLM
Let's try SimToM with an example from the ToMI dataset:


story = """
1 Aria entered the front_yard.
2 Aiden entered the front_yard.
3 The grapefruit is in the green_bucket.
4 Aria moved the grapefruit to the blue_container.
5 Aiden exited the front_yard.
6 Noah entered the playroom.
"""
question = "7 Where was the grapefruit at the beginning?"
character = "Aria"
We load Mistral-7B-Instruct-v0.3, create the prompt using the template we defined earlier, and generate a structured response. As a reminder, the goal of the first call is to get all the events a character, Aria, knows about.


# Load an LLM from Hugging Face
MODEL_NAME = "mistral-community/Mistral-7B-Instruct-v0.3"
model = outlines.models.transformers(MODEL_NAME, device="cuda")

perspective_prompt = perspective_taking(story=story, character=character)

# Call Mistral 7B with the first prompt
generator = outlines.generate.json(model, PerspectiveTaking)
perspective = generator(perspective_prompt)

print(perspective.model_dump())
# {'character': 'Aria', 'events': ['1 Aria entered the front_yard.', '3 The grapefruit is in the green_bucket.', '4 Aria moved the grapefruit to the blue_container.']}
Not bad! We will now generate the second prompt with those events.


sim_prompt = simulation(events=perspective.events, name=character, question=question)

# Call Mistral 7B with the second prompt
generator = outlines.generate.json(model, Simulation)
result = generator(sim_prompt)

print(result.model_dump())
# {'answer': 'green_bucket'}
And this is it! SimToM could be useful in agentic workflows, where agents must act based on what they know, not all available information. One caveat of SimToM is that the perspective-taking step may remove important information, leading to wrong results. As the authors note in their paper, it can feature as a simple and effective baseline for evaluating LLMs on Theory of Mind reasoning tasks.

Generate Synthetic Data and Q&A with Citations
This tutorial is adapted from the instructor-ollama notebook. We start with a simple example to generate synthetic data and then we approach the problem of question answering by providing citations.

We will use llama.cpp using the llama-cpp-python library. Outlines supports llama-cpp-python, but we need to install it ourselves:


pip install llama-cpp-python
We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
(Optional) Store the model weights in a custom folder
Generate Synthetic Data
We first need to define our Pydantic class for a user:


from pydantic import BaseModel, Field

class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier") # so the model keeps track of the number of users
    first_name: str
    last_name: str
    age: int
We then define a Pydantic class for a list of users:


from typing import List

class Users(BaseModel):
    users: List[UserDetail]
We can use a generate.json by passing this Pydantic class we just defined, and call the generator:


model = models.LlamaCpp(llm)
generator = generate.json(model, Users)
response = generator("Create 5 fake users", max_tokens=1024, temperature=0, seed=42)
print(response.users)
# [UserDetail(id=1, first_name='John', last_name='Doe', age=25),
# UserDetail(id=2, first_name='Jane', last_name='Doe', age=30),
# UserDetail(id=3, first_name='Bob', last_name='Smith', age=40),
# UserDetail(id=4, first_name='Alice', last_name='Smith', age=35),
# UserDetail(id=5, first_name='John', last_name='Smith', age=20)]

for user in response.users:
    print(user.first_name)
    print(user.last_name)
    print(user.age)
    print(#####)
# John
# Doe
# 25
# #####
# Jane
# Doe
# 30
# #####
# Bob
# Smith
# 40
# #####
# Alice
# Smith
# 35
# #####
# John
# Smith
# 20
# #####
QA with Citations
We first need to define our Pydantic class for QA with citations:


from typing import List
from pydantic import BaseModel

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    citations: List[str]

schema = QuestionAnswer.model_json_schema()
We then need to adapt our prompt to the Hermes prompt format for JSON schema:


def generate_hermes_prompt(question, context, schema=schema):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON with correct and exact citations "
        "extracted from the `Context`. "
        f"Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>\n"
        "<|im_start|>user\n"
        + "`Context`: "
        + context
        + "\n`Question`: "
        + question + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )
We can use generate.json by passing the Pydantic class we previously defined, and call the generator with Hermes prompt:


question = "What did the author do during college?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
I went to an arts high school but in university I studied Computational Mathematics and physics.
As part of coop I worked at many companies including Stitchfix, Facebook.
I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""
generator = generate.json(model, QuestionAnswer)
prompt = generate_hermes_prompt(question, context)
response = generator(prompt, max_tokens=1024, temperature=0, seed=42)
print(response)
# QuestionAnswer(question='What did the author do during college?', answer='The author studied Computational Mathematics and physics in university and was also involved in starting the Data Science club, serving as its president for 2 years.', citations=['I went to an arts high school but in university I studied Computational Mathematics and physics.', 'I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.'])
We can do the same for a list of question-context pairs:


question1 = "Where was John born?"
context1 = """
John Doe is a software engineer who was born in New York, USA.
He studied Computer Science at the Massachusetts Institute of Technology.
During his studies, he interned at Google and Microsoft.
He also founded the Artificial Intelligence club at his university and served as its president for three years.
"""

question2 = "What did Emily study in university?"
context2 = """
Emily Smith is a data scientist from London, England.
She attended the University of Cambridge where she studied Statistics and Machine Learning.
She interned at IBM and Amazon during her summer breaks.
Emily was also the head of the Women in Tech society at her university.
"""

question3 = "Which companies did Robert intern at?"
context3 = """
Robert Johnson, originally from Sydney, Australia, is a renowned cybersecurity expert.
He studied Information Systems at the University of Melbourne.
Robert interned at several cybersecurity firms including NortonLifeLock and McAfee.
He was also the leader of the Cybersecurity club at his university.
"""

question4 = "What club did Alice start at her university?"
context4 = """
Alice Williams, a native of Dublin, Ireland, is a successful web developer.
She studied Software Engineering at Trinity College Dublin.
Alice interned at several tech companies including Shopify and Squarespace.
She started the Web Development club at her university and was its president for two years.
"""

question5 = "What did Michael study in high school?"
context5 = """
Michael Brown is a game developer from Tokyo, Japan.
He attended a specialized high school where he studied Game Design.
He later attended the University of Tokyo where he studied Computer Science.
Michael interned at Sony and Nintendo during his university years.
He also started the Game Developers club at his university.
"""

for question, context in [
    (question1, context1),
    (question2, context2),
    (question3, context3),
    (question4, context4),
    (question5, context5),
]:
    final_prompt = my_final_prompt(question, context)
    generator = generate.json(model, QuestionAnswer)
    response = generator(final_prompt, max_tokens=1024, temperature=0, seed=42)
    display(question)
    display(response.answer)
    display(response.citations)
    print("\n\n")

# 'Where was John born?'
# 'John Doe was born in New York, USA.'
# ['John Doe is a software engineer who was born in New York, USA.']
#
#
# 'What did Emily study in university?'
# 'Emily studied Statistics and Machine Learning in university.'
# ['She attended the University of Cambridge where she studied Statistics and Machine Learning.']
#
#
# 'Which companies did Robert intern at?'
# 'Robert interned at NortonLifeLock and McAfee.'
# ['Robert Johnson, originally from Sydney, Australia, is a renowned cybersecurity expert. He interned at several cybersecurity firms including NortonLifeLock and McAfee.']
#
#
# 'What club did Alice start at her university?'
# 'Alice started the Web Development club at her university.'
# ['Alice Williams, a native of Dublin, Ireland, is a successful web developer. She started the Web Development club at her university and was its president for two years.']
#
#
# 'What did Michael study in high school?'
# 'Michael studied Game Design in high school.'
# ['Michael Brown is a game developer from Tokyo, Japan. He attended a specialized high school where he studied Game Design.']

Knowledge Graph Extraction
In this guide, we use outlines to extract a knowledge graph from unstructured text.

We will use llama.cpp using the llama-cpp-python library. Outlines supports llama-cpp-python, but we need to install it ourselves:


pip install llama-cpp-python
We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
(Optional) Store the model weights in a custom folder
Knowledge Graph Extraction
We first need to define our Pydantic class for each node and each edge of the knowledge graph:


from pydantic import BaseModel, Field

class Node(BaseModel):
    """Node of the Knowledge Graph"""

    id: int = Field(..., description="Unique identifier of the node")
    label: str = Field(..., description="Label of the node")
    property: str = Field(..., description="Property of the node")


class Edge(BaseModel):
    """Edge of the Knowledge Graph"""

    source: int = Field(..., description="Unique source of the edge")
    target: int = Field(..., description="Unique target of the edge")
    label: str = Field(..., description="Label of the edge")
    property: str = Field(..., description="Property of the edge")
We then define the Pydantic class for the knowledge graph and get its JSON schema:


from typing import List

class KnowledgeGraph(BaseModel):
    """Generated Knowledge Graph"""

    nodes: List[Node] = Field(..., description="List of nodes of the knowledge graph")
    edges: List[Edge] = Field(..., description="List of edges of the knowledge graph")

schema = KnowledgeGraph.model_json_schema()
We then need to adapt our prompt to the Hermes prompt format for JSON schema:


def generate_hermes_prompt(user_prompt):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON "
        f"Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>\n"
        "<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )
For a given user prompt, for example:


user_prompt = "Alice loves Bob and she hates Charlie."
We can use generate.json by passing the Pydantic class we previously defined, and call the generator with the Hermes prompt:


from outlines import generate, models

model = models.LlamaCpp(llm)
generator = generate.json(model, KnowledgeGraph)
prompt = generate_hermes_prompt(user_prompt)
response = generator(prompt, max_tokens=1024, temperature=0, seed=42)
We obtain the nodes and edges of the knowledge graph:


print(response.nodes)
print(response.edges)
# [Node(id=1, label='Alice', property='Person'),
# Node(id=2, label='Bob', property='Person'),
# Node(id=3, label='Charlie', property='Person')]
# [Edge(source=1, target=2, label='love', property='Relationship'),
# Edge(source=1, target=3, label='hate', property='Relationship')]
(Optional) Visualizing the Knowledge Graph
We can use the Graphviz library to visualize the generated knowledge graph. For detailed installation instructions, see here.


from graphviz import Digraph

dot = Digraph()
for node in response.nodes:
    dot.node(str(node.id), node.label, shape='circle', width='1', height='1')
for edge in response.edges:
    dot.edge(str(edge.source), str(edge.target), label=edge.label)

dot.render('knowledge-graph.gv', view=True)


Chain of thought
Chain of thought is a prompting technique introduced in the paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" where throught prompting the authors generate a series of intermediate reasoning steps which improves the ability of LLMs to perform complex reasoning.

In this guide, we use outlines to apply chain of thought through structured output.

We use llama.cpp using the llama-cpp-python library. Outlines supports llama-cpp-python, but we need to install it ourselves:


pip install llama-cpp-python
We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
(Optional) Store the model weights in a custom folder
Chain of thought
We first define our Pydantic class for a reasoning step:


from pydantic import BaseModel, Field

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="Reasoning step")
We then define the Pydantic class for reasoning which will consist on a list of reasoning steps and a conclusion, and we get its JSON schema:


from typing import List

class Reasoning(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: str = Field(..., description="Conclusion")

json_schema = Reasoning.model_json_schema()
We could generate a response using the json schema but for a change we will use the regex:


from outlines.integrations.utils import convert_json_schema_to_str
from outlines.fsm.json_schema import build_regex_from_schema

schema_str = convert_json_schema_to_str(json_schema=json_schema)
regex_str = build_regex_from_schema(schema_str)
We then need to adapt our prompt to the Hermes prompt format for JSON schema:


def generate_hermes_prompt(user_prompt):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON "
        f"Here's the json schema you must adhere to:\n<schema>\n{json_schema}\n</schema><|im_end|>\n"
        "<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )
For a given user prompt:


user_prompt = "9.11 and 9.9 -- which is bigger?"
we can use generate.regex by passing the Pydantic class we previously defined, and call the generator with the Hermes prompt:


generator = generate.regex(model, regex_str)
prompt = generate_hermes_prompt(user_prompt)
response = generator(prompt, max_tokens=1024, temperature=0, seed=42)
We obtain a series of intermediate reasoning steps as well as the conclusion:


import json

json_response = json.loads(response)

print(json_response["reasoning"])
print(json_response["conclusion"])
# [{'reasoning_step': 'Both 9.11 and 9.9 are decimal numbers.'},
#  {'reasoning_step': 'When comparing decimal numbers, we look at the numbers after the decimal point.'},
#  {'reasoning_step': 'In this case, 9.11 has the number 1 after the decimal point, while 9.9 has the number 9.'},
#  {'reasoning_step': 'Since 1 is greater than 9, 9.11 is greater than 9.9.'}]
# '9.11 is bigger.'


ReAct Agent
This example shows how to use outlines to build your own agent with open weights local models and structured outputs. It is inspired by the blog post A simple Python implementation of the ReAct pattern for LLMs by Simon Willison.

The ReAct pattern (for Reason+Act) is described in the paper ReAct: Synergizing Reasoning and Acting in Language Models. It's a pattern where you implement additional actions that an LLM can take - searching Wikipedia or running calculations for example - and then teach it how to request the execution of those actions, and then feed their results back into the LLM.

Additionally, we give the LLM the possibility of using a scratchpad described in the paper Show Your Work: Scratchpads for Intermediate Computation with Language Models which improves the ability of LLMs to perform multi-step computations.

We use llama.cpp using the llama-cpp-python library. Outlines supports llama-cpp-python, but we need to install it ourselves:


pip install llama-cpp-python
We download the model weights by passing the name of the repository on the HuggingFace Hub, and the filenames (or glob pattern):


import llama_cpp
from outlines import generate, models

model = models.llamacpp("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
            "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
            tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
            ),
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=8192,
            verbose=False)
(Optional) Store the model weights in a custom folder
Build a ReAct agent
In this example, we use two tools:

wikipedia: \<search term> - search Wikipedia and returns the snippet of the first result
calculate: \<expression> - evaluate an expression using Python's eval() function

import httpx

def wikipedia(q):
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]


def calculate(numexp):
    return eval(numexp)
We define the logic of the agent through a Pydantic class. First, we want the LLM to decide only between the two previously defined tools:


from enum import Enum

class Action(str, Enum):
    wikipedia = "wikipedia"
    calculate = "calculate"
Our agent will loop through Thought and Action. We explicitly give the Action Input field so it doesn't forget to add the arguments of the Action. We also add a scratchpad (optional).


from pydantic import BaseModel, Field

class Reason_and_Act(BaseModel):
    Scratchpad: str = Field(..., description="Information from the Observation useful to answer the question")
    Thought: str = Field(..., description="It describes your thoughts about the question you have been asked")
    Action: Action
    Action_Input: str = Field(..., description="The arguments of the Action.")
Our agent will reach a Final Answer. We also add a scratchpad (optional).


class Final_Answer(BaseModel):
    Scratchpad: str = Field(..., description="Information from the Observation useful to answer the question")
    Final_Answer: str = Field(..., description="Answer to the question grounded on the Observation")
Our agent will decide when it has reached a Final Answer and therefore to stop the loop of Thought and Action.


from typing import Union

class Decision(BaseModel):
    Decision: Union[Reason_and_Act, Final_Answer]
We could generate a response using the json schema but we will use the regex and check that everything is working as expected:


from outlines.integrations.utils import convert_json_schema_to_str
from outlines.fsm.json_schema import build_regex_from_schema

json_schema = Decision.model_json_schema()
schema_str = convert_json_schema_to_str(json_schema=json_schema)
regex_str = build_regex_from_schema(schema_str)
print(regex_str)
# '\\{[ ]?"Decision"[ ]?:[ ]?(\\{[ ]?"Scratchpad"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Thought"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Action"[ ]?:[ ]?("wikipedia"|"calculate")[ ]?,[ ]?"Action_Input"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?\\}|\\{[ ]?"Scratchpad"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?,[ ]?"Final_Answer"[ ]?:[ ]?"([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\])*"[ ]?\\})[ ]?\\}'
We then need to adapt our prompt to the Hermes prompt format for JSON schema and explain the agent logic:


import datetime

def generate_hermes_prompt(question, schema=""):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON with correct Pydantic schema. "
        f"Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>\n"
        "Today is " + datetime.datetime.today().strftime('%Y-%m-%d') + ".\n" +
        "You run in a loop of Scratchpad, Thought, Action, Action Input, PAUSE, Observation. "
        "At the end of the loop you output a Final Answer. "
        "Use Scratchpad to store the information from the Observation useful to answer the question "
        "Use Thought to describe your thoughts about the question you have been asked "
        "and reflect carefully about the Observation if it exists. "
        "Use Action to run one of the actions available to you. "
        "Use Action Input to input the arguments of the selected action - then return PAUSE. "
        "Observation will be the result of running those actions. "
        "Your available actions are:\n"
        "calculate:\n"
        "e.g. calulate: 4**2 / 3\n"
        "Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n"
        "wikipedia:\n"
        "e.g. wikipedia: Django\n"
        "Returns a summary from searching Wikipedia\n"
        "DO NOT TRY TO GUESS THE ANSWER. Begin! <|im_end|>"
        "\n<|im_start|>user\n" + question + "<|im_end|>"
        "\n<|im_start|>assistant\n"
    )
We define a ChatBot class


class ChatBot:
    def __init__(self, prompt=""):
        self.prompt = prompt

    def __call__(self, user_prompt):
        self.prompt += user_prompt
        result = self.execute()
        return result

    def execute(self):
        generator = generate.regex(model, regex_str)
        result = generator(self.prompt, max_tokens=1024, temperature=0, seed=42)
        return result
We define a query function:


import json

def query(question, max_turns=5):
    i = 0
    next_prompt = (
        "\n<|im_start|>user\n" + question + "<|im_end|>"
        "\n<|im_start|>assistant\n"
    )
    previous_actions = []
    while i < max_turns:
        i += 1
        prompt = generate_hermes_prompt(question=question, schema=Decision.model_json_schema())
        bot = ChatBot(prompt=prompt)
        result = bot(next_prompt)
        json_result = json.loads(result)['Decision']
        if "Final_Answer" not in list(json_result.keys()):
            scratchpad = json_result['Scratchpad'] if i == 0 else ""
            thought = json_result['Thought']
            action = json_result['Action']
            action_input = json_result['Action_Input']
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Thought: {thought} \x1b[0m")
            print(f"\x1b[36m  -- running {action}: {str(action_input)}\x1b[0m")
            if action + ": " + str(action_input) in previous_actions:
                observation = "You already run that action. **TRY A DIFFERENT ACTION INPUT.**"
            else:
                if action=="calculate":
                    try:
                        observation = eval(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
                elif action=="wikipedia":
                    try:
                        observation = wikipedia(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
            print()
            print(f"\x1b[33m Observation: {observation} \x1b[0m")
            print()
            previous_actions.append(action + ": " + str(action_input))
            next_prompt += (
                "\nScratchpad: " + scratchpad +
                "\nThought: " + thought +
                "\nAction: " + action  +
                "\nAction Input: " + action_input +
                "\nObservation: " + str(observation)
            )
        else:
            scratchpad = json_result["Scratchpad"]
            final_answer = json_result["Final_Answer"]
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Final Answer: {final_answer} \x1b[0m")
            return final_answer
    print(f"\nFinal Answer: I am sorry, but I am unable to answer your question. Please provide more information or a different question.")
    return "No answer found"
We can now test our ReAct agent:


print(query("What's 2 to the power of 10?"))
# Scratchpad:
# Thought: I need to perform a mathematical calculation to find the result of 2 to the power of 10.
#  -- running calculate: 2**10
#
# Observation: 1024
#
# Scratchpad: 2 to the power of 10 is 1024.
# Final Answer: 2 to the power of 10 is 1024.
# 2 to the power of 10 is 1024.

print(query("What does England share borders with?"))
# Scratchpad:
# Thought: To answer this question, I will use the 'wikipedia' action to gather information about England's geographical location and its borders.
#  -- running wikipedia: England borders
#
# Observation: Anglo-Scottish <span class="searchmatch">border</span> (Scottish Gaelic: Crìochan Anglo-Albannach) is an internal <span class="searchmatch">border</span> of the United Kingdom separating Scotland and <span class="searchmatch">England</span> which runs for
#
# Scratchpad: Anglo-Scottish border (Scottish Gaelic: Crìochan Anglo-Albannach) is an internal border of the United Kingdom separating Scotland and England which runs for
# Final Answer: England shares a border with Scotland.
# England shares a border with Scotland.
As mentioned in Simon's blog post, this is not a very robust implementation at all and there's a ton of room for improvement. But it is lovely how simple it is with a few lines of Python to make these extra capabilities available to the LLM. And now you can run it locally with an open weights LLM.