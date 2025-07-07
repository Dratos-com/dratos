"""
* WORKING

What this script does:
Reads reviews from a CSV file, uses an LLM to classify the type of 
traffic infraction mentioned (if any) based on a predefined category structure,
and saves the enriched data to a new CSV file.

Requirements:
- pandas: `pip install pandas`
- Add the folowing API key(s) in your .env file:
  - GOOGLE_API_KEY (or OPENAI_API_KEY if using OpenAI)
"""

################### Adding Project Root to Python Path #############################
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)


import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dratos import LLM, Agent, GoogleEngine

# Add Project Root to Python Path (assuming the script is run from the project root)
# If running from `examples/`, adjust the path accordingly.
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

################### Adding Project Root to Python Path #############################

# Define the infraction categories
INFRACTION_CATEGORIES = {
  "Vitesse": {
    "Excès de vitesse": [
      "< 20 km/h (limite > 50 km/h)",
      "< 20 km/h (limite ≤ 50 km/h)",
      "20 à 29 km/h au-dessus",
      "30 à 39 km/h au-dessus",
      "40 à 49 km/h au-dessus",
      "≥ 50 km/h au-dessus"
    ]
  },
  "Signalisation": {
    "Feu rouge grillé": [],
    "Non-respect d'un stop": []
  },
  "Comportement": {
    "Non-port de la ceinture de sécurité": [],
    "Téléphone tenu en main": [],
    "Distance de sécurité insuffisante": [],
    "Circulation en sens interdit": [],
    "Demi-tour ou marche arrière sur autoroute": [],
    "Dépassement dangereux/interdit": [],
    "Refus de faciliter un dépassement": [],
    "Blocage d'intersection": [],
    "Circulation voie de gauche (hors dépassement)": []
  },
  "Stationnement/Circulation": {
    "Voie réservée (bus, taxis...)": [],
    "Bande d'arrêt d'urgence": [],
    "Chevauchement ligne continue": [],
    "Franchissement ligne continue": []
  },
  "Équipement/Documents": {
    "Non-port du casque (2-roues)": [],
    "Bruit excessif du véhicule": [],
    "Défaut d'assurance": []
  },
  "Autre": {
      "Non spécifié": []
  }
}

# Generate lists for Pydantic validation
VALID_CATEGORIES = list(INFRACTION_CATEGORIES.keys())
VALID_INFRACTIONS = [infraction for category_data in INFRACTION_CATEGORIES.values() for infraction in category_data.keys()]
VALID_GRAVITIES = list(set(gravity for category_data in INFRACTION_CATEGORIES.values() for gravities in category_data.values() for gravity in gravities))
VALID_GRAVITIES.append("Non applicable") # Add a default/fallback

# Define the structured output model
class InfractionResult(BaseModel):
    category: Optional[Literal[tuple(VALID_CATEGORIES)]] = Field(
        None, description="The general category of the infraction based on the review text."
    )
    infraction: Optional[Literal[tuple(VALID_INFRACTIONS)]] = Field(
        None, description="The specific infraction mentioned in the review text."
    )
    gravity: Optional[Literal[tuple(VALID_GRAVITIES)]] = Field(
        None, description="The specific gravity or detail of the infraction, if mentioned (e.g., speed range)."
    )

# Configure the LLM and Agent
llm = LLM(
    model_name="gemini-2.0-flash", # Use a capable model
    engine=GoogleEngine(),
)

classification_agent = Agent(
    name="infraction_classifier",
    verbose=True, # Set to False for cleaner output during batch processing
    llm=llm,
    response_model=InfractionResult,
    response_validation=True,
)

def classify_review(review_text: str) -> InfractionResult:
    """
    Uses an LLM agent to classify a review text based on predefined infraction categories.
    """
    if not review_text or pd.isna(review_text):
        return InfractionResult() # Return empty result for empty input

    # Translate the prompt to French and clarify the None/null instruction
    prompt = f"""
    Analysez le texte d'avis suivant concernant un avocat spécialisé en droit routier.
    Votre objectif est d'identifier si l'avis mentionne explicitement une infraction routière spécifique pour laquelle l'avocat est intervenu.
    Utilisez UNIQUEMENT les catégories, infractions et gravités fournies ci-dessous.

    Structure des Catégories :
    {INFRACTION_CATEGORIES}

    Texte de l'avis :
    "{review_text}"

    Instructions :
    1. Lisez attentivement le texte de l'avis.
    2. Déterminez s'il mentionne une infraction routière spécifique listée dans la Structure des Catégories.
    3. Si une infraction spécifique est mentionnée (par exemple, "feu rouge", "excès de vitesse", "stop non respecté", "ligne continue"), identifiez la 'category' et l''infraction' correspondantes.
    4. Si l'infraction mentionnée a une gravité spécifique qui correspond à une gravité listée (par exemple, "excès de vitesse de plus de 50 km/h"), identifiez la 'gravity'. Si aucune gravité spécifique n'est mentionnée mais que l'infraction est claire, utilisez 'Non applicable' pour la gravité.
    5. Si l'avis est un éloge général (par exemple, "très professionnel", "a sauvé mon permis") ou mentionne une perte de points/permis sans spécifier l'*infraction exacte* de la liste, OU s'il mentionne une infraction NON présente dans la liste (comme 'voie de bus' qui n'est pas explicitement 'Voie réservée'), définissez 'category' sur 'Autre', 'infraction' sur 'Non spécifié', et 'gravity' sur 'Non applicable'.
    6. Si le texte de l'avis n'est pas clair, trop vague, ou s'il ne mentionne aucune infraction spécifique identifiable selon la liste fournie, retournez `null` (ou `None` en Python) pour les champs 'category', 'infraction', et 'gravity'. Ne forcez pas une classification si elle n'est pas évidente.

    Retournez le résultat au format JSON spécifié avec les champs : 'category', 'infraction', 'gravity'. Assurez-vous que les valeurs retournées correspondent *exactement* à celles fournies dans la Structure des Catégories ou sont `null`.
    """
    try:
        result = classification_agent.sync_gen({"text": prompt})
        # Basic check if the result is a dict-like object before parsing
        if isinstance(result, dict):
            # Handle potential direct dict return or Pydantic model
            # Replace potential 'null' strings or empty strings with None before validation
            cleaned_result = {k: (v if v not in [None, '', 'null', 'Null', 'None'] else None) for k, v in result.items()}
            return InfractionResult(**cleaned_result)
        elif isinstance(result, InfractionResult):
             # Ensure None values are handled correctly if the model comes back directly
             result_dict = result.dict()
             cleaned_result = {k: (v if v not in [None, '', 'null', 'Null', 'None'] else None) for k, v in result_dict.items()}
             return InfractionResult(**cleaned_result)
        else:
             print(f"Warning: Unexpected result type: {type(result)}. Returning empty.")
             return InfractionResult() # Return Pydantic model with all fields as None

    except Exception as e:
        print(f"Error processing review: {review_text[:50]}... Error: {e}")
        return InfractionResult() # Return empty result on error

def main():
    """
    Main function to read CSV, classify reviews, and save the enriched CSV.
    """
    input_csv_path = "examples/test.csv"
    output_csv_path = "examples/test_enriched.csv"

    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at {input_csv_path}")
        return

    print(f"Reading CSV file from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Ensure 'caption' column exists
    if 'caption' not in df.columns:
        print(f"Error: 'caption' column not found in {input_csv_path}")
        return

    print(f"Processing {len(df)} reviews...")

    results = []
    # Using apply for potentially cleaner iteration
    # Ensure review_text passed to classify_review is a string
    df['caption'] = df['caption'].astype(str)

    # *** Process sequentially ***
    processed_count = 0
    total_rows = len(df)
    for review in df['caption']:
        print(f"Processing review {processed_count + 1}/{total_rows}...")
        result = classify_review(review)
        results.append(result)
        processed_count += 1
        # Optional: Add a small delay if hitting API rate limits
        # import time
        # time.sleep(0.5) # Sleep for 500ms

    # Convert results (list of Pydantic models) into DataFrame columns
    df_results = pd.DataFrame([r.dict() for r in results])

    # Add new columns to the original DataFrame
    df['category'] = df_results['category']
    df['infraction'] = df_results['infraction']
    df['gravity'] = df_results['gravity']

    print(f"\nSaving enriched data to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print("Processing complete.")

if __name__ == "__main__":
    main()