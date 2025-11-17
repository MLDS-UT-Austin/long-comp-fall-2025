"""
Generate embeddings for NLP Meeting agent
This will create the question_data_with_embeddings.pkl file
"""

# add ".." to path for imports
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import *
from util import *

if __name__ == "__main__":
    embedding = GeminiEmbedding()
    question_data = pd.read_csv("example agents/all_question_bank.csv")

    # Create embeddings for all questions
    print("Generate embeddings for questions")
    print("This should take <10 seconds...")

    question_data["question_embedding"] = asyncio.get_event_loop().run_until_complete(
        embedding.get_embeddings(question_data["question"].tolist())
    )

    print("Generate embeddings for answers")
    print("This should take <10 seconds...")

    question_data["answer_embedding"] = asyncio.get_event_loop().run_until_complete(
        embedding.get_embeddings(question_data["answer"].tolist())
    )

    question_data.to_pickle("example agents/question_data_with_embeddings.pkl")
    print("Saved to example agents/question_data_with_embeddings.pkl")