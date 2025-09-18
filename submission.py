from agent import Agent, register_agent
from data import Location, redaction_dict
from nlp import LLMRole, NLPProxy
from util import redact

"""
Team Member Names: 
Team Member Emails: 
Team Member EIDs: 
"""


@register_agent("Team Name Here")
class MyAgent(Agent):
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

    async def ask_question(self) -> tuple[int, str]:
        return 1, "question"

    async def answer_question(self, question: str) -> str:
        return "answer"

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        pass

    async def guess_location(self) -> Location | None:
        return None

    async def accuse_player(self) -> int | None:
        return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        pass


# Validate agent
MyAgent.validate()
