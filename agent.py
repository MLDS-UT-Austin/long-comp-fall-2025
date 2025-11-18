import asyncio
from abc import ABC, abstractmethod
from typing import final

from data import Location
from nlp import LLMRole, NLPProxy
from util import relative_path_decorator
from tqdm import tqdm

# This file explains what each method in the Agent class should do ######################

AGENT_REGISTRY = {}


def register_agent(team_name: str):
    """Type @register_agent("Team Name Here") on top of the class definition to register the agent"""

    def decorator(cls):
        assert issubclass(cls, Agent)
        assert team_name not in AGENT_REGISTRY
        # this changes all methods in the class to use paths relative to the class file
        cls = relative_path_decorator(cls)
        AGENT_REGISTRY[team_name] = cls
        return cls

    return decorator


class Agent(ABC):
    """This is the base class for all agents
    You should subclass this class and implement the abstract methods in submission.py
    Note: Throughout this class, players are indexed from 0 to n_players - 1, inclusive with player 0 being you
          also, locations are represented by the Location enum
    """

    @abstractmethod
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        """Args:
        location (Location | None): the enum location for the game or None if the agent is the spy
        n_players (int): number of players including yourself
        n_rounds (int): total number rounds. Each round includes a question, answer, and a vote.
        nlp (NLPProxy): allows you to prompt the llm and get embeddings. You are given a set number of tokens per round.
            If you exceed the token limit, get_prompt will return an empty string and get_embeddings will return a 0 array.
        """
        pass

    @abstractmethod
    async def ask_question(self) -> tuple[int, str]:
        """This method is called when it is the agent's turn to ask a question.

        Returns:
            tuple[int, str]: the index of the opponent to ask the question and the question to ask
                Note: opponent indices range from 1 to n_players - 1, inclusive
        """
        return 1, "question here"

    @abstractmethod
    async def answer_question(self, question: str) -> str:
        """This method is called when the agent is asked a question.

        Args:
            question (str): the question asked by the opponent

        Returns:
            str: your answer to the question
        """
        return "answer here"

    @abstractmethod
    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        """This method is called every round after a question is asked and answered.

        Args:
            questioner (int): the index of the opponent who asked the question
            question (str): the question asked by the opponent
            answerer (int): the index of the opponent who answered the question
            answer (str): the response to the question
        """
        pass

    @abstractmethod
    async def guess_location(self) -> Location | None:
        """This method is called every round when the agent is the spy

        Returns:
            Location | None: The location guessed by the agent or None if the agent does not want to guess
                Note: The agent can only guess once per game
        """
        return Location.BLANTON_MUSEUM

    @abstractmethod
    async def accuse_player(self) -> int | None:
        """This method is called at the end of every round

        Returns:
            int | None: The index of the opponent the agent accuses of being the spy or None if the agent does not want to accuse
        """
        return 1

    @abstractmethod
    async def analyze_voting(self, votes: list[int | None]) -> None:
        """This method is called at the end of every round after all players have voted
        Args:
            votes (list[int  |  None]): a list containing the vote for each player
                Ex: [0, None, ...] means player 0 voted for you, player 1 did not vote, etc.
        """
        pass

    @classmethod
    @final
    def validate(cls) -> None:
        """Quick check for return types and edge cases"""
        event_loop = asyncio.get_event_loop()
        nlp = NLPProxy()
        for _ in tqdm(range(100)):
            for loc, n_players in [
                (Location.BLANTON_MUSEUM, 3),
                (None, 3),
                (Location.HARRY_RANSOM_CENTER, 10),
                (None, 10),
            ]:
                agent = cls(loc, n_players, 5, nlp)
                answerer, question = event_loop.run_until_complete(agent.ask_question())
                answer0 = event_loop.run_until_complete(
                    agent.answer_question("question here")
                )
                answer1 = event_loop.run_until_complete(agent.answer_question(""))
                event_loop.run_until_complete(
                    agent.analyze_response(1, "question here", 2, "answer here")
                )
                event_loop.run_until_complete(
                    agent.analyze_response(0, "question here", 2, "answer here")
                )
                event_loop.run_until_complete(agent.analyze_response(2, "", 0, ""))
                guess = event_loop.run_until_complete(agent.guess_location())
                accusation = event_loop.run_until_complete(agent.accuse_player())
                event_loop.run_until_complete(agent.analyze_voting([0, 1, None, None]))
                assert isinstance(answerer, int)
                assert 1 <= answerer < n_players
                assert isinstance(question, str)
                assert isinstance(answer0, str)
                assert isinstance(answer1, str)
                assert isinstance(guess, Location) or guess is None
                assert isinstance(accusation, int) or accusation is None
                if accusation is not None:
                    assert 1 <= accusation < n_players
