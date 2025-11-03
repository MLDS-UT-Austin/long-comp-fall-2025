import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="google.api_core._python_version_support"
)
warnings.filterwarnings(
    "ignore",
    message="This feature is deprecated as of June 24, 2025",
    category=UserWarning,
    module=r"vertexai\.generative_models\._generative_models",
)

warnings.filterwarnings(
    "ignore",
    message="This feature is deprecated as of June 24, 2025",
    category=UserWarning,
    module=r"vertexai\._model_garden\._model_garden_models",
)

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part, ThinkingConfig


# Set this to True to print out all LLM calls for debugging
LLM_VERBOSE = False

# Adjust these as needed if you want to change your GC project or location
GOOGLE_CLOUD_PROJECT = "long-comp-fall-2025"
GOOGLE_CLOUD_LOCATION = "us-central1"

# This file explains how LLMRole and NLPProxy work, which are used by your agent to interact with the LLM #################


class LLMRole(Enum):
    USER = "user"
    MODEL = "model"


class NLPProxy:
    """The wrapper that agents will use to interact with the LLM"""

    def __init__(self, token_counter=None):
        if token_counter is None:
            token_counter = TokenCounterWrapper()
        self.__token_counter = token_counter

    async def prompt_llm(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        """prompts the LLM with a given prompt and returns the output

        Args:
            prompt (list[tuple[LLMRole, str]]): List of tuples containing the role and text.
                Use LLMRole.USER to give the llm the prompt and any background information.
                LLMRole.MODEL can be used to give the llm an example of how to respond.

            max_output_tokens (int | None, optional): The maximum number of tokens to output or None for no limit

            temperature (float, optional): The temperature to use for the llm. A higher temperature produces more varied results. Defaults to 0.7.

        Returns:
            str: llm output
        """
        return await self.__token_counter.prompt_llm(
            prompt, max_output_tokens, temperature
        )

    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        """Gets a 768-dimensional embedding for the given text

        Args:
            text (str): Input text. This will counted against the token limit but at a lesser extent than the llm.
                Every 10 tokens inputted into the embedding model is equivalent to 1 token inputted into the llm.

        Returns:
            np.ndarray: 768-dimensional embedding
        """
        return await self.__token_counter.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        """Used to count the number of tokens that would be used by the llm. This can help agents manage their token usage.

        Args:
            text_or_prompt (str | list[tuple[LLMRole, str]]): Input text as a string or a list of tuples containing the role and text

        Returns:
            int: Number of tokens
        """
        return self.__token_counter.count_llm_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str | list[str]) -> int:
        """Equivalent to count_llm_tokens but for the embedding model"""
        return self.__token_counter.count_embedding_tokens(text)

    def get_remaining_tokens(self) -> int:
        """
        Returns:
            int: the number of tokens remaining for your agent for the round
        """
        return self.__token_counter.remaining_tokens


# Everything below is for internal use only ####################################################################

GENAI_IS_INITIALIZED = False


def initialize_genai():
    global GENAI_IS_INITIALIZED
    global client
    if not GENAI_IS_INITIALIZED:
        client = genai.Client(
            vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION
        )
        GENAI_IS_INITIALIZED = True


# Interfaces ####################################################################


class LLMTokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        pass


class LLM(ABC):
    @abstractmethod
    async def prompt(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        pass


class EmbeddingTokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text: str | list[str]) -> int:
        pass


class Embedding(ABC):
    @abstractmethod
    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        pass


# Implementations ####################################################################


class GeminiTokenizer(LLMTokenizer):
    def __init__(self):
        initialize_genai()
        self.model_name = "gemini-2.5-flash"

    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        if isinstance(text_or_prompt, list):
            contents = [
                Content(role=role.value, parts=[Part(text=text)])
                for role, text in text_or_prompt
            ]
        else:
            if text_or_prompt == "":
                return 0
            contents = text_or_prompt

        token_info = client.models.count_tokens(
            model=self.model_name, contents=contents
        )
        return token_info.total_tokens


class GeminiLLM(LLM):
    def __init__(self):
        initialize_genai()
        self.model_name = "gemini-2.5-flash"

    async def prompt(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        # Convert role/content tuples to GenAI Message objects
        contents = [
            Content(role=role.value, parts=[Part(text=text)]) for role, text in prompt
        ]

        # Use asyncio.to_thread to significantly speed up blocking calls
        def generate_sync():
            global client
            return client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_config=ThinkingConfig(thinking_budget=0),
                ),
            )

        response = await asyncio.to_thread(generate_sync)
        
        # uncomment for debugging prompt/response
        # concat_prompt = " ".join([f"{role.value}: {text}" for role, text in prompt])
        # print(f"\tinput: {concat_prompt}\noutput: {response.text}")

        # uncomment for debugging token usage
        # print(
        #     f"input tokens: {response.usage_metadata.prompt_token_count}, "
        #     f"thinking tokens: {response.usage_metadata.thoughts_token_count}, "
        #     f"max_output_tokens: {max_output_tokens}, "
        #     f"output: {response.text}"
        # )
        return response.text


class DummyLLM(LLM):
    """A dummy LLM for testing"""

    async def prompt(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        return "Output from the LLM will be here"


class GeminiEmbeddingTokenizer(EmbeddingTokenizer):
    # set location to "us-central1"
    def __init__(self):
        initialize_genai()
        # gemini-embedding-001 does not work, but we can estimate with gemini-2.5-flash
        # self.model_name = "gemini-embedding-001"
        self.model_name = "gemini-2.5-flash"

    def count_tokens(self, text: str | list[str]) -> int:
        global client
        texts = [text] if isinstance(text, str) else text

        # Filter out empty strings
        non_empty_texts = [t for t in texts if t != ""]

        if not non_empty_texts:
            return 0

        combined_text = " ".join(non_empty_texts)
        token_info = client.models.count_tokens(
            model=self.model_name, contents=combined_text
        )
        return token_info.total_tokens


class GeminiEmbedding(Embedding):
    def __init__(self):
        initialize_genai()
        self.model_name = "gemini-embedding-001"

    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        """Returns embeddings using Vertex AI's textembedding-gecko model."""
        texts = [text] if isinstance(text, str) else text

        # Use asyncio.to_thread to significantly speed up blocking calls
        def get_embeddings_sync():
            # Use batch processing - embed_content can handle a list of texts
            response = client.models.embed_content(
                model=self.model_name, contents=texts
            )
            # Extract embeddings from response
            embeddings = [np.array(emb.values) for emb in response.embeddings]
            return embeddings

        embeddings = await asyncio.to_thread(get_embeddings_sync)

        if isinstance(text, str):
            return embeddings[0]
        return embeddings


class DummyEmbedding(Embedding):
    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        if isinstance(text, str):
            return np.zeros(3072)  # Updated to match Gemini embedding dimension
        return [np.zeros(3072) for _ in text]


@dataclass
class NLP:
    """Used for the single, main NLP instance in the runtime"""

    llm_tokenizer: LLMTokenizer = field(default_factory=GeminiTokenizer)
    llm: LLM = field(default_factory=DummyLLM)
    embedding_tokenizer: EmbeddingTokenizer = field(
        default_factory=GeminiEmbeddingTokenizer
    )
    embedding: Embedding = field(default_factory=DummyEmbedding)

    async def prompt_llm(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        return await self.llm.prompt(prompt, max_output_tokens, temperature)

    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        return await self.embedding.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        return self.llm_tokenizer.count_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str | list[str]) -> int:
        return self.embedding_tokenizer.count_tokens(text)

    # disable pickling
    def __getstate__(self):
        return {}


@dataclass
class TokenCounterWrapper:
    """Used for NLP instances that need to keep track of token usage"""

    nlp: NLP = field(default_factory=NLP)
    token_limit: int = 1000
    player_name: str | None = None

    def __post_init__(self):
        self.reset_token_counter()

    def reset_token_counter(self):
        self.remaining_tokens = self.token_limit

    async def prompt_llm(
        self,
        prompt: list[tuple[LLMRole, str]],
        max_output_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        if self.remaining_tokens is not None:
            self.remaining_tokens -= self.nlp.count_llm_tokens(prompt)
            if self.remaining_tokens <= 0:
                return "<out of tokens>"

            if max_output_tokens is None:
                max_output_tokens = self.remaining_tokens
            else:
                max_output_tokens = min(max_output_tokens, self.remaining_tokens)

            output = await self.nlp.prompt_llm(prompt, max_output_tokens, temperature)
            self.remaining_tokens -= self.nlp.count_llm_tokens(output)

            if self.remaining_tokens <= 0:
                output += " <out of tokens>"

        if LLM_VERBOSE:
            prompt_text = " ".join(
                [f"{role.value}: {text}" for role, text in prompt]
            )
            player_info = (
                f" for player {self.player_name}" if self.player_name else ""
            )
            print(
                f"[LLM call{player_info}]\n\tInput: {prompt_text}\n\tOutput: {output}\n"
            )


        return output

    async def get_embeddings(
        self, text: str | list[str]
    ) -> np.ndarray | list[np.ndarray]:
        self.remaining_tokens -= math.ceil(self.nlp.count_embedding_tokens(text) / 10)
        if self.remaining_tokens < 0:
            if isinstance(text, str):
                return np.zeros(3072)
            return [np.zeros(3072) for _ in text]

        return await self.nlp.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        return self.nlp.count_llm_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str | list[str]) -> int:
        return self.nlp.count_embedding_tokens(text)


if __name__ == "__main__":
    # run this to test all the components
    event_loop = asyncio.get_event_loop()

    llm_tokenizer = GeminiTokenizer()
    tokens = llm_tokenizer.count_tokens("How many US states are there?")
    print(tokens)

    llm = GeminiLLM()
    prompt = [(LLMRole.USER, "How many US states are there?")]
    output = event_loop.run_until_complete(llm.prompt(prompt, 20))
    print(output)

    embedding_tokenizer = GeminiEmbeddingTokenizer()
    tokens = embedding_tokenizer.count_tokens("How many US states are there?")
    print(tokens)

    tokens = embedding_tokenizer.count_tokens(["How many", "US states are there?"])
    print(tokens)

    embedding_model = GeminiEmbedding()
    output = event_loop.run_until_complete(
        embedding_model.get_embeddings("How many US states are there?")
    )
    print(f"first 5: {output[:5]}")
    print(type(output))
    print(len(output))

    output = event_loop.run_until_complete(
        embedding_model.get_embeddings(["How many", "US states are there?"])
    )
    print(type(output))
    print(len(output))

    output = event_loop.run_until_complete(
        embedding_model.get_embeddings(["How many", "US states are there?"])
    )
    print(type(output))
    print(len(output))

    embedding_model = DummyEmbedding()
    output = event_loop.run_until_complete(
        embedding_model.get_embeddings("How many US states are there?")
    )
    print(f"first 5: {output[:5]}")
    print(type(output))
    print(len(output))

    output = event_loop.run_until_complete(
        embedding_model.get_embeddings(["How many", "US states are there?"])
    )
    print(type(output))
    print(len(output))
    print(type(output))
    print(len(output))
