"""
Example demonstrating the LiteLLM engine with multiple providers.

This example shows how to use LiteLLM to access different LLM providers
through a unified interface while leveraging DRATOS's agent framework.
"""

import os
from dratos import Agent, LiteLLMEngine, LLM


def basic_litellm_example():
    """Basic example using LiteLLM with OpenAI (default provider)."""
    print("=== Basic LiteLLM Example (OpenAI) ===")
    
    # Create LiteLLM engine with OpenAI as provider
    engine = LiteLLMEngine(
        api_key=os.getenv("OPENAI_API_KEY"),  # or set via environment
        provider="openai"
    )
    
    # Create LLM instance
    llm = LLM(
        model_name="gpt-4o",
        engine=engine
    )
    
    # Create agent
    agent = Agent(
        name="litellm_assistant",
        llm=llm,
        verbose=True
    )
    
    response = agent.sync_gen({"text": "Explain the benefits of using LiteLLM in 2 sentences."})
    print(f"Response: {response}")
    return response


def anthropic_example():
    """Example using LiteLLM with Anthropic's Claude."""
    print("\n=== LiteLLM with Anthropic Claude ===")
    
    # Create LiteLLM engine with Anthropic as provider
    engine = LiteLLMEngine(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        provider="anthropic"
    )
    
    llm = LLM(
        model_name="claude-3-sonnet-20240229",  # LiteLLM handles provider routing
        engine=engine
    )
    
    agent = Agent(
        name="claude_assistant",
        llm=llm,
        verbose=True
    )
    
    response = agent.sync_gen({"text": "What are the key differences between traditional API wrappers and LiteLLM?"})
    print(f"Claude Response: {response}")
    return response


def multi_provider_example():
    """Example showing how to easily switch between providers."""
    print("\n=== Multi-Provider Example ===")
    
    providers_config = [
        {
            "name": "OpenAI GPT-4o",
            "engine": LiteLLMEngine(provider="openai"),
            "model": "gpt-4o"
        },
        {
            "name": "Vertex AI Gemini", 
            "engine": LiteLLMEngine(provider="vertex_ai"),
            "model": "vertex_ai/gemini-1.5-flash"
        },
        {
            "name": "Claude-3 Sonnet", 
            "engine": LiteLLMEngine(provider="anthropic"),
            "model": "claude-3-sonnet-20240229"
        },
        # Add more providers as needed
    ]
    
    prompt = "What is the capital of France?"
    
    for config in providers_config:
        try:
            print(f"\n--- Testing {config['name']} ---")
            
            llm = LLM(
                model_name=config['model'],
                engine=config['engine']
            )
            
            agent = Agent(
                name=f"agent_{config['name'].lower().replace(' ', '_')}",
                llm=llm
            )
            
            response = agent.sync_gen({"text": prompt})
            print(f"{config['name']} Response: {response}")
            
        except Exception as e:
            print(f"Error with {config['name']}: {e}")


def vertex_ai_example():
    """Example using LiteLLM with Vertex AI Gemini."""
    print("\n=== Vertex AI Gemini Example ===")
    
    # Option 1: Using environment variables (recommended - same as GoogleEngine)
    # Set these in your environment:
    # export GOOGLE_CLOUD_PROJECT="your-project-id"
    # export GOOGLE_CLOUD_REGION="us-central1"
    # export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    
    engine = LiteLLMEngine(provider="vertex_ai")
    
    # Option 2: Explicit configuration (uncomment and modify)
    # engine = LiteLLMEngine(
    #     provider="vertex_ai",
    #     vertex_project="your-project-id",        # or use GOOGLE_CLOUD_PROJECT env var
    #     vertex_location="us-central1",           # or use GOOGLE_CLOUD_REGION env var
    #     vertex_credentials="/path/to/service.json"  # or use GOOGLE_APPLICATION_CREDENTIALS env var
    # )
    
    llm = LLM(
        model_name="vertex_ai/gemini-1.5-flash",  # Use vertex_ai/ prefix
        engine=engine
    )
    
    agent = Agent(
        name="vertex_agent",
        llm=llm,
        verbose=True
    )
    
    try:
        response = agent.sync_gen({
            "text": "Explain the benefits of using Vertex AI for ML workloads in 2 sentences."
        })
        print(f"Vertex AI Response: {response}")
        return response
    except Exception as e:
        print(f"Error with Vertex AI: {e}")
        print("Make sure you have configured Vertex AI credentials and project settings.")
        print("Required env vars: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION, GOOGLE_APPLICATION_CREDENTIALS")
        return None


def vertex_ai_tools_example():
    """Example showing Vertex AI Gemini with tools."""
    print("\n=== Vertex AI Tools Example ===")
    
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"The weather in {location} is sunny and 72°F"
    
    def calculate_price(base_price: float, tax_rate: float) -> float:
        """Calculate total price with tax."""
        return base_price * (1 + tax_rate)
    
    engine = LiteLLMEngine(provider="vertex_ai")
    
    llm = LLM(
        model_name="vertex_ai/gemini-1.5-flash",
        engine=engine
    )
    
    agent = Agent(
        name="vertex_tool_agent",
        llm=llm,
        system_prompt="You are a helpful assistant with access to weather and pricing tools.",
        tools=[get_weather, calculate_price],
        verbose=True
    )
    
    try:
        response = agent.sync_gen({
            "text": "What's the weather in Tokyo and what would be the total cost of a $50 item with 8% tax?"
        })
        print(f"Vertex AI Tools Response: {response}")
        return response
    except Exception as e:
        print(f"Vertex AI Tools example failed: {e}")
        return None


def vertex_ai_vision_example():
    """Example showing Vertex AI Gemini with vision capabilities."""
    print("\n=== Vertex AI Vision Example ===")
    
    engine = LiteLLMEngine(provider="vertex_ai")
    
    llm = LLM(
        model_name="vertex_ai/gemini-1.5-flash",  # Gemini models support vision
        engine=engine
    )
    
    agent = Agent(
        name="vertex_vision_agent",
        llm=llm,
        system_prompt="You are a helpful assistant that can analyze images.",
        verbose=True
    )
    
    try:
        # Test with a single image URL
        response = agent.sync_gen({
            "text": "What's in this image? Describe it briefly.",
            "image.jpg": "https://ichef.bbci.co.uk/images/ic/1920xn/p072ms6r.jpg"
        })
        print(f"Vertex AI Vision Response: {response}")
        return response
    except Exception as e:
        print(f"Vertex AI Vision example failed: {e}")
        return None


def vertex_ai_multi_image_example():
    """Example showing Vertex AI Gemini with multiple images."""
    print("\n=== Vertex AI Multi-Image Example ===")
    
    engine = LiteLLMEngine(provider="vertex_ai")
    
    llm = LLM(
        model_name="vertex_ai/gemini-1.5-flash",
        engine=engine
    )
    
    agent = Agent(
        name="vertex_multi_image_agent",
        llm=llm,
        system_prompt="You are a helpful assistant that can analyze and compare multiple images.",
        verbose=True
    )
    
    try:
        # Test with multiple images
        response = agent.sync_gen({
            "text": "Compare these two images. What are the main differences?",
            "image1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/CatLolCatExample.jpg/170px-CatLolCatExample.jpg",
            "image2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Farmer_meme_with_apostrophe.jpg/220px-Farmer_meme_with_apostrophe.jpg"
        })
        print(f"Vertex AI Multi-Image Response: {response}")
        return response
    except Exception as e:
        print(f"Vertex AI Multi-Image example failed: {e}")
        return None


def vertex_ai_vision_base64_example():
    """Example showing Vertex AI Gemini with base64 encoded image."""
    print("\n=== Vertex AI Vision Base64 Example ===")
    
    engine = LiteLLMEngine(provider="vertex_ai")
    
    llm = LLM(
        model_name="vertex_ai/gemini-1.5-flash",
        engine=engine
    )
    
    agent = Agent(
        name="vertex_base64_agent",
        llm=llm,
        system_prompt="You are a helpful assistant that can analyze images provided in base64 format.",
        verbose=True
    )
    
    try:
        # Convert image URL to base64
        import base64
        from urllib.request import urlopen
        import ssl

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/CatLolCatExample.jpg/170px-CatLolCatExample.jpg"
        ssl._create_default_https_context = ssl._create_unverified_context
        image_data = urlopen(image_url).read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        response = agent.sync_gen({
            "text": "What's the subject of this image? Describe it in one sentence.",
            "cat_meme.jpeg": base64_image
        })
        print(f"Vertex AI Base64 Vision Response: {response}")
        return response
    except Exception as e:
        print(f"Vertex AI Base64 Vision example failed: {e}")
        return None


def tools_example():
    """Example using LiteLLM with tools/function calling."""
    print("\n=== LiteLLM Tools Example ===")
    
    def calculate_price(base_price: float, tax_rate: float = 0.1) -> float:
        """Calculate total price including tax."""
        return base_price * (1 + tax_rate)
    
    def get_weather(city: str) -> str:
        """Get weather information for a city."""
        # Mock weather function
        return f"The weather in {city} is sunny, 75°F"
    
    # Create LiteLLM engine
    engine = LiteLLMEngine(provider="openai")  # Fixed: was using vertex_ai but model was gpt-4o
    
    llm = LLM(
        model_name="gpt-4o",
        engine=engine
    )
    
    agent = Agent(
        name="tool_agent",
        llm=llm,
        tools=[calculate_price, get_weather],
        verbose=True
    )
    
    response = agent.sync_gen({
        "text": "What's the weather in Paris and what would be the total cost of a $100 item with 15% tax?"
    })
    print(f"Tools Response: {response}")
    return response


def custom_endpoint_example():
    """Example using LiteLLM with a custom OpenAI-compatible endpoint."""
    print("\n=== Custom Endpoint Example ===")
    
    # Example with a local or custom OpenAI-compatible server
    engine = LiteLLMEngine(
        base_url="http://localhost:11434/v1",  # Example: Ollama server
        provider="openai",
        api_key="ollama"  # Some local servers require any key
    )
    
    llm = LLM(
        model_name="llama2",  # Model available on your custom endpoint
        engine=engine
    )
    
    agent = Agent(
        name="custom_endpoint_agent",
        llm=llm
    )
    
    try:
        response = agent.sync_gen({"text": "Hello from custom endpoint!"})
        print(f"Custom Endpoint Response: {response}")
        return response
    except Exception as e:
        print(f"Custom endpoint not available: {e}")
        return None


if __name__ == "__main__":
    # Set up environment variables (or use .env file)
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    
    print("LiteLLM Engine Examples")
    print("=" * 50)
    
    # Run examples
    # try:
    #     basic_litellm_example()
    # except Exception as e:
    #     print(f"Basic example failed: {e}")
    
    try:
        vertex_ai_example()
    except Exception as e:
        print(f"Vertex AI example failed: {e}")
    
    try:
        vertex_ai_tools_example()
    except Exception as e:
        print(f"Vertex AI tools example failed: {e}")
    
    try:
        vertex_ai_vision_example()
    except Exception as e:
        print(f"Vertex AI vision example failed: {e}")
    
    try:
        vertex_ai_multi_image_example()
    except Exception as e:
        print(f"Vertex AI multi-image example failed: {e}")
    
    try:
        vertex_ai_vision_base64_example()
    except Exception as e:
        print(f"Vertex AI base64 vision example failed: {e}")
    
    # try:
    #     tools_example()
    # except Exception as e:
    #     print(f"Tools example failed: {e}")
    
    # try:
    #     anthropic_example()
    # except Exception as e:
    #     print(f"Anthropic example failed: {e}")
    
    # try:
    #     multi_provider_example()
    # except Exception as e:
    #     print(f"Multi-provider example failed: {e}")
    
    # try:
    #     custom_endpoint_example()
    # except Exception as e:
    #     print(f"Custom endpoint example failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")