from beta import Agents, Models, Tools, Data
import mlflow

# Initialize MLflow client
mlflow_client = mlflow.tracking.MlflowClient()

# Initialize the VLLM engine with a large language model
engine = Models.VLLMEngine(
    model_name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    mlflow_client=mlflow_client
)

# Get the language model
language_model = Models.get(
    model_name="gpt-3.5-turbo",
    engine=engine,
    mlflow_client=mlflow_client
)

# Initialize tools
sentiment_tool = Tools.get("SentimentAnalysisTool")
entity_extraction_tool = Tools.get("EntityExtractionTool")
summarization_tool = Tools.get("SummarizationTool")
knowledge_graph_tool = Tools.get("KnowledgeGraphTool")
qa_tool = Tools.get("QuestionAnsweringTool")

# Create an agent with the model and tools
agent = Agents.TextAnalysisAgent(
    model=language_model,
    tools=[
        sentiment_tool,
        entity_extraction_tool,
        summarization_tool,
        knowledge_graph_tool,
        qa_tool
    ]
)

# Create a knowledge base to store analysis results
knowledge_base = Data.KnowledgeBase()

def analyze_document(document):
    # Task 1: Sentiment Analysis
    sentiment = agent.use_tool(sentiment_tool, document)
    knowledge_base.add("sentiment", sentiment)

    # Task 2: Entity Extraction
    entities = agent.use_tool(entity_extraction_tool, document)
    knowledge_base.add("entities", entities)

    # Task 3: Summarization
    summary = agent.use_tool(summarization_tool, document)
    knowledge_base.add("summary", summary)

    # Task 4: Knowledge Graph Creation
    knowledge_graph = agent.use_tool(knowledge_graph_tool, entities)
    knowledge_base.add("knowledge_graph", knowledge_graph)

    # Task 5: Question Answering
    def answer_question(question):
        return agent.use_tool(qa_tool, {
            "question": question,
            "context": document,
            "knowledge_base": knowledge_base
        })

    return {
        "sentiment": sentiment,
        "entities": entities,
        "summary": summary,
        "knowledge_graph": knowledge_graph,
        "answer_question": answer_question
    }

# Example usage
if __name__ == "__main__":
    document = """
    Artificial Intelligence (AI) is revolutionizing various industries. 
    Machine Learning, a subset of AI, enables systems to learn from data. 
    Natural Language Processing allows computers to understand human language. 
    These technologies are driving innovation in fields like healthcare, 
    finance, and transportation.
    """
    
    results = analyze_document(document)

    print("Document Summary:", results["summary"])
    print("Overall Sentiment:", results["sentiment"])
    print("Key Entities:", results["entities"])
    print("Knowledge Graph:", results["knowledge_graph"])

    # Ask questions about the document
    question = "What are the main applications of AI mentioned in the document?"
    answer = results["answer_question"](question)
    print("Q:", question)
    print("A:", answer)
