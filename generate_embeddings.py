"""
Generate embeddings for NLP Meeting agent
This will create the question_data_with_embeddings.pkl file
"""

from simulation import *
from util import *

if __name__ == "__main__":
    print("=" * 80)
    print("Generating Embeddings for NLP Meeting Agent")
    print("=" * 80)
    print()
    
    # Use Gemini embedding
    nlp = NLP(llm=GeminiLLM(), embedding=GeminiEmbedding())
    
    # Load the NLP Meeting agent - this will trigger embedding generation
    print("Loading NLP Meeting agent...")
    print("(This will generate embeddings for 185 questions and answers)")
    print("(This may take 2-3 minutes)")
    print()
    
    import_agents_from_files("example agents/agents.py")
    
    # Instantiate the agent to trigger __init__
    from agent import AGENT_REGISTRY
    from data import Location
    
    print("Creating NLP Meeting agent instance...")
    agent = AGENT_REGISTRY["NLP Meeting"](
        location=Location.BLANTON_MUSEUM_OF_ART,  # Dummy location for testing
        n_players=4,
        n_rounds=20,
        nlp=nlp
    )
    
    print()
    print("=" * 80)
    print("✓ Embeddings generated successfully!")
    print("=" * 80)
    print()
    print("File created: example agents/question_data_with_embeddings.pkl")
    print(f"Total questions/answers: {len(agent.question_data)}")
    print(f"Embedding dimension: {agent.question_data['question_embedding'].iloc[0].shape}")
