
"""
TODO progress

What this script does:
Generate a structured response from an LLM.

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

from dratos.memory.obj.artifacts import Artifact
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")

# Where my artifacts will be stored
artifact_repository = f"my_artifacts/{str(today)}/"

# Create my artifacts
papers = Artifact(files=[
    "examples/data/Papers/2401.05459v2.pdf",
    "examples/data/Papers/2401.18059.pdf",
    "examples/data/Papers/2402.01680v2.pdf",
    "examples/data/Papers/2402.01968v1.pdf"
], uri_prefix=artifact_repository)

br_23_earnings = Artifact(files=[
    "examples/data/Blackrock_23_Earnings/BLK 1Q23 Earnings Release.pdf",
    "examples/data/Blackrock_23_Earnings/BLK 2Q23 Earnings Release.pdf",
    "examples/data/Blackrock_23_Earnings/BLK 3Q23 Earnings Release.pdf",
    "examples/data/Blackrock_23_Earnings/BLK 4Q23 Earnings Release.pdf"
], uri_prefix=artifact_repository)

print(papers.uri_prefix)
print(br_23_earnings.uri_prefix)