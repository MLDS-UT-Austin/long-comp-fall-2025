import asyncio
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from agent_data import *

from agent import Agent, register_agent
from data import Location, redaction_dict
from nlp import GeminiEmbedding, LLMRole, NLPProxy
from util import redact

ACCUSE_PROB = 0.9
ACCUSE_NOISE = 0.05
GUESS_THRESHOLD = 1.0  # was 0.5


@register_agent("MLDS 0")
class MLDS0(Agent):
    """Example agent that uses multi-shot prompting to generate answers and analyze responses."""

    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        # shared data
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

        self.spy = location is None
        self.answerer_count = np.zeros(n_players - 1, dtype=int)
        self.round = 0
        self.est_n_rounds_remaining = n_rounds
        self.last_questioner = -1
        self.most_voted_player = random.randint(1, n_players - 1)
        self.question_bank = QUESTIONS.copy()

        # spy data
        self.avg_loc_score = {loc: 0.0 for loc in Location}

        # nonspy data
        self.avg_spy_score = np.zeros(n_players - 1, dtype=float)

    async def ask_question(self) -> tuple[int, str]:
        if self.spy:
            # Give those who haven't been asked a higher chance of being asked
            answerer_score = -self.answerer_count + np.random.rand() * 0.1
        else:
            # it's likely that this is the last question we will ask, so question the spy
            if self.est_n_rounds_remaining < self.n_players:
                answerer_score = self.avg_spy_score
            # there will likely be another chance to ask a question, so explore the other players
            else:
                answerer_score = (
                    self.avg_spy_score
                    + 1 / (self.answerer_count + 1)
                    + np.random.rand() * 0.1
                )
        # set the last questioner to be the least likely to be asked
        if self.last_questioner != -1:
            answerer_score[self.last_questioner - 1] = -np.inf
        answerer = int(np.argmax(answerer_score)) + 1
        # now, select the question
        # ensure questions are not repeated when possible
        if len(self.question_bank) == 0:
            self.question_bank = QUESTIONS.copy()
        question = self.question_bank.pop(
            random.randint(0, len(self.question_bank) - 1)
        )
        return answerer, question

    async def _answer_question_spy(self, question: str) -> str:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.ASSISTANT, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def answer_question(self, question: str) -> str:
        question = question[:200]  # limit question length
        if self.spy:
            return await self._answer_question_spy(question)
        else:
            return await self._answer_question_nonspy(question)

    @staticmethod
    def _get_locs_from_str(loc_str: str) -> list[Location]:
        output = []
        for loc in Location:
            if loc.value.lower() in loc_str.lower().strip():
                output.append(loc)
        return output

    @staticmethod
    def _get_float_from_str(float_str: str) -> float | None:
        try:
            return float(float_str)
        except:
            return None

    async def _analyze_response_spy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"Do you see animals here?\" Answer: \"Yes, turtles and fish are common.\""),
            (LLMRole.ASSISTANT, "Turtle Pond, Lake Austin"),
            (LLMRole.USER, f"Question: \"Do people take photos here?\" Answer: \"Yes\""),
            (LLMRole.ASSISTANT, "Blanton Museum of Art, UT Tower, Littlefield Fountain, Turtle Pond, Texas State Capitol, Mt. Bonnell & Mayfield Park, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"Is the architecture grande?\" Answer: \"Yes, it is impressive and historic.\""),
            (LLMRole.ASSISTANT, "Texas State Capitol, UT Tower, Littlefield Fountain, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 72, 0.25)
        locs = self._get_locs_from_str(reponse)
        for loc in Location:
            if loc in locs:
                self.avg_loc_score[loc] += 1 / len(locs)
            else:
                self.avg_loc_score[loc] -= 1 / (len(Location) - len(locs))

    async def _analyze_response_nonspy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 10, 0.25)
        prob = self._get_float_from_str(reponse)
        if prob is not None:
            # Averaging spy scores per player
            if self.answerer_count[answerer - 1] == 1:
                self.avg_spy_score[answerer - 1] += prob
            else:
                # Recalculate weighted average
                current_count = self.answerer_count[answerer - 1]
                self.avg_spy_score[answerer - 1] = (
                    self.avg_spy_score[answerer - 1] * (current_count - 1) + prob
                ) / current_count

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        self.last_questioner = questioner
        if answerer == 0:
            return
        question = question[:200]  # limit question length
        answer = answer[:200]  # limit answer length
        self.answerer_count[answerer - 1] += 1
        if self.spy:
            await self._analyze_response_spy(questioner, question, answerer, answer)
        else:
            await self._analyze_response_nonspy(questioner, question, answerer, answer)

    async def guess_location(self) -> Location | None:
        # if the top location score is 0.5 higher than the second highest, guess that location
        loc_scores = list(self.avg_loc_score.items())
        loc_scores.sort(key=lambda x: x[1], reverse=True)
        if loc_scores[0][1] - loc_scores[1][1] > GUESS_THRESHOLD:
            return loc_scores[0][0]
        return None

    async def accuse_player(self) -> int | None:
        if self.spy:
            if self.most_voted_player == 0:
                return random.randint(1, self.n_players - 1)
            return self.most_voted_player
        else:
            if (
                self.avg_spy_score[np.argmax(self.avg_spy_score)] >= 0.5
                and random.random() < ACCUSE_PROB
            ):
                score = self.avg_spy_score + ACCUSE_NOISE * np.random.normal(
                    0, 1, size=(self.n_players - 1)
                )
                player = int(np.argmax(self.avg_spy_score)) + 1
                return player
            return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        self.round += 1
        percent_voted = sum(vote is not None for vote in votes) / len(votes)
        self.est_n_rounds_remaining = int(
            (self.n_rounds - self.round) * (1 - percent_voted * 2)
        )
        c = Counter(votes)
        del c[None]
        if len(c) > 0:
            self.most_voted_player = c.most_common(1)[0][0]  # type: ignore


@register_agent("MLDS 1")
class MLDS1(Agent):
    """Same as MLDS 0, but more agressive in accusing players of being the spy."""

    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        # shared data
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

        self.spy = location is None
        self.answerer_count = np.zeros(n_players - 1, dtype=int)
        self.round = 0
        self.est_n_rounds_remaining = n_rounds
        self.last_questioner = -1
        self.most_voted_player = random.randint(1, n_players - 1)
        self.question_bank = QUESTIONS.copy()

        # spy data
        self.avg_loc_score = {loc: 0.0 for loc in Location}

        # nonspy data
        self.avg_spy_score = np.zeros(n_players - 1, dtype=float)

    async def ask_question(self) -> tuple[int, str]:
        if self.spy:
            # Give those who haven't been asked a higher chance of being asked
            answerer_score = -self.answerer_count + np.random.rand() * 0.1
        else:
            # it's likely that this is the last question we will ask, so question the spy
            if self.est_n_rounds_remaining < self.n_players:
                answerer_score = self.avg_spy_score
            # there will likely be another chance to ask a question, so explore the other players
            else:
                answerer_score = (
                    self.avg_spy_score
                    + 1 / (self.answerer_count + 1)
                    + np.random.rand() * 0.1
                )
        # set the last questioner to be the least likely to be asked
        if self.last_questioner != -1:
            answerer_score[self.last_questioner - 1] = -np.inf
        answerer = int(np.argmax(answerer_score)) + 1
        # now, select the question
        # ensure questions are not repeated when possible
        if len(self.question_bank) == 0:
            self.question_bank = QUESTIONS.copy()
        question = self.question_bank.pop(
            random.randint(0, len(self.question_bank) - 1)
        )
        return answerer, question

    async def _answer_question_spy(self, question: str) -> str:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.ASSISTANT, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def answer_question(self, question: str) -> str:
        question = question[:200]  # limit question length
        if self.spy:
            return await self._answer_question_spy(question)
        else:
            return await self._answer_question_nonspy(question)

    @staticmethod
    def _get_locs_from_str(loc_str: str) -> list[Location]:
        output = []
        for loc in Location:
            if loc.value.lower() in loc_str.lower().strip():
                output.append(loc)
        return output

    @staticmethod
    def _get_float_from_str(float_str: str) -> float | None:
        try:
            return float(float_str)
        except:
            return None

    async def _analyze_response_spy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. List of only the most reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"Do you see animals here?\" Answer: \"Yes, turtles and fish are common.\""),
            (LLMRole.ASSISTANT, "Turtle Pond, Lake Austin"),
            (LLMRole.USER, f"Question: \"Do people take photos here?\" Answer: \"Yes\""),
            (LLMRole.ASSISTANT, "Blanton Museum of Art, UT Tower, Littlefield Fountain, Turtle Pond, Texas State Capitol, Mt. Bonnell & Mayfield Park, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"Is the architecture grande?\" Answer: \"Yes, it is impressive and historic.\""),
            (LLMRole.ASSISTANT, "Texas State Capitol, UT Tower, Littlefield Fountain, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 72, 0.25)
        locs = self._get_locs_from_str(reponse)
        for loc in Location:
            if loc in locs:
                self.avg_loc_score[loc] += 1 / len(locs)
            else:
                self.avg_loc_score[loc] -= 1 / (len(Location) - len(locs))

    async def _analyze_response_nonspy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Be a bit passive about your guesses, but keep in mind people giving vauge answers are more likely to be the spy. The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 10, 0.25)
        prob = self._get_float_from_str(reponse)
        if prob is not None:
            # Averaging spy scores per player
            if self.answerer_count[answerer - 1] == 1:
                self.avg_spy_score[answerer - 1] += prob
            else:
                # Recalculate weighted average
                current_count = self.answerer_count[answerer - 1]
                self.avg_spy_score[answerer - 1] = (
                    self.avg_spy_score[answerer - 1] * (current_count - 1) + prob
                ) / current_count

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        self.last_questioner = questioner
        if answerer == 0:
            return
        question = question[:200]  # limit question length
        answer = answer[:200]  # limit answer length
        self.answerer_count[answerer - 1] += 1
        if self.spy:
            await self._analyze_response_spy(questioner, question, answerer, answer)
        else:
            await self._analyze_response_nonspy(questioner, question, answerer, answer)

    async def guess_location(self) -> Location | None:
        # if the top location score is 0.5 higher than the second highest, guess that location
        loc_scores = list(self.avg_loc_score.items())
        loc_scores.sort(key=lambda x: x[1], reverse=True)
        if loc_scores[0][1] - loc_scores[1][1] > GUESS_THRESHOLD:
            return loc_scores[0][0]
        return None

    async def accuse_player(self) -> int | None:
        if self.spy:
            if self.most_voted_player == 0:
                return random.randint(1, self.n_players - 1)
            return self.most_voted_player
        else:
            if (
                self.avg_spy_score[np.argmax(self.avg_spy_score)] >= 0.5
                and random.random() < ACCUSE_PROB
            ):
                score = self.avg_spy_score + ACCUSE_NOISE * np.random.normal(
                    0, 1, size=(self.n_players - 1)
                )
                player = int(np.argmax(self.avg_spy_score)) + 1
                return player
            return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        self.round += 1
        percent_voted = sum(vote is not None for vote in votes) / len(votes)
        self.est_n_rounds_remaining = int(
            (self.n_rounds - self.round) * (1 - percent_voted * 2)
        )
        c = Counter(votes)
        del c[None]
        if len(c) > 0:
            self.most_voted_player = c.most_common(1)[0][0]  # type: ignore


@register_agent("MLDS 2")
class MLDS2(Agent):
    """Same as MLDS 0, but less agressive in accusing players of being the spy."""

    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        # shared data
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

        self.spy = location is None
        self.answerer_count = np.zeros(n_players - 1, dtype=int)
        self.round = 0
        self.est_n_rounds_remaining = n_rounds
        self.last_questioner = -1
        self.most_voted_player = random.randint(1, n_players - 1)
        self.question_bank = QUESTIONS.copy()

        # spy data
        self.avg_loc_score = {loc: 0.0 for loc in Location}

        # nonspy data
        self.avg_spy_score = np.zeros(n_players - 1, dtype=float)

    async def ask_question(self) -> tuple[int, str]:
        if self.spy:
            # Give those who haven't been asked a higher chance of being asked
            answerer_score = -self.answerer_count + np.random.rand() * 0.1
        else:
            # it's likely that this is the last question we will ask, so question the spy
            if self.est_n_rounds_remaining < self.n_players:
                answerer_score = self.avg_spy_score
            # there will likely be another chance to ask a question, so explore the other players
            else:
                answerer_score = (
                    self.avg_spy_score
                    + 1 / (self.answerer_count + 1)
                    + np.random.rand() * 0.1
                )
        # set the last questioner to be the least likely to be asked
        if self.last_questioner != -1:
            answerer_score[self.last_questioner - 1] = -np.inf
        answerer = int(np.argmax(answerer_score)) + 1
        # now, select the question
        # ensure questions are not repeated when possible
        if len(self.question_bank) == 0:
            self.question_bank = QUESTIONS.copy()
        question = self.question_bank.pop(
            random.randint(0, len(self.question_bank) - 1)
        )
        return answerer, question

    async def _answer_question_spy(self, question: str) -> str:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.ASSISTANT, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.ASSISTANT, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 20, 0.25)
        return answer

    async def answer_question(self, question: str) -> str:
        question = question[:200]  # limit question length
        if self.spy:
            return await self._answer_question_spy(question)
        else:
            return await self._answer_question_nonspy(question)

    @staticmethod
    def _get_locs_from_str(loc_str: str) -> list[Location]:
        output = []
        for loc in Location:
            if loc.value.lower() in loc_str.lower().strip():
                output.append(loc)
        return output

    @staticmethod
    def _get_float_from_str(float_str: str) -> float | None:
        try:
            return float(float_str)
        except:
            return None

    async def _analyze_response_spy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"Do you see animals here?\" Answer: \"Yes, turtles and fish are common.\""),
            (LLMRole.ASSISTANT, "Turtle Pond, Lake Austin"),
            (LLMRole.USER, f"Question: \"Do people take photos here?\" Answer: \"Yes\""),
            (LLMRole.ASSISTANT, "Blanton Museum of Art, UT Tower, Littlefield Fountain, Turtle Pond, Texas State Capitol, Mt. Bonnell & Mayfield Park, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"Is the architecture grande?\" Answer: \"Yes, it is impressive and historic.\""),
            (LLMRole.ASSISTANT, "Texas State Capitol, UT Tower, Littlefield Fountain, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 72, 0.25)
        locs = self._get_locs_from_str(reponse)
        for loc in Location:
            if loc in locs:
                self.avg_loc_score[loc] += 1 / len(locs)
            else:
                self.avg_loc_score[loc] -= 1 / (len(Location) - len(locs))

    async def _analyze_response_nonspy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Keep in mind people giving vauge answers are more likely to be the spy. Be somewhat aggressive in your guesses here. We wanna catch the spy quickly! The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 10, 0.25)
        prob = self._get_float_from_str(reponse)
        if prob is not None:
            # Averaging spy scores per player
            if self.answerer_count[answerer - 1] == 1:
                self.avg_spy_score[answerer - 1] += prob
            else:
                # Recalculate weighted average
                current_count = self.answerer_count[answerer - 1]
                self.avg_spy_score[answerer - 1] = (
                    self.avg_spy_score[answerer - 1] * (current_count - 1) + prob
                ) / current_count

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        self.last_questioner = questioner
        if answerer == 0:
            return
        question = question[:200]  # limit question length
        answer = answer[:200]  # limit answer length
        self.answerer_count[answerer - 1] += 1
        if self.spy:
            await self._analyze_response_spy(questioner, question, answerer, answer)
        else:
            await self._analyze_response_nonspy(questioner, question, answerer, answer)

    async def guess_location(self) -> Location | None:
        # if the top location score is 0.5 higher than the second highest, guess that location
        loc_scores = list(self.avg_loc_score.items())
        loc_scores.sort(key=lambda x: x[1], reverse=True)
        if loc_scores[0][1] - loc_scores[1][1] > GUESS_THRESHOLD:
            return loc_scores[0][0]
        return None

    async def accuse_player(self) -> int | None:
        if self.spy:
            if self.most_voted_player == 0:
                return random.randint(1, self.n_players - 1)
            return self.most_voted_player
        else:
            if (
                self.avg_spy_score[np.argmax(self.avg_spy_score)] >= 0.5
                and random.random() < ACCUSE_PROB
            ):
                score = self.avg_spy_score + ACCUSE_NOISE * np.random.normal(
                    0, 1, size=(self.n_players - 1)
                )
                player = int(np.argmax(self.avg_spy_score)) + 1
                return player
            return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        self.round += 1
        percent_voted = sum(vote is not None for vote in votes) / len(votes)
        self.est_n_rounds_remaining = int(
            (self.n_rounds - self.round) * (1 - percent_voted * 2)
        )
        c = Counter(votes)
        del c[None]
        if len(c) > 0:
            self.most_voted_player = c.most_common(1)[0][0]  # type: ignore


@register_agent("NLP Meeting")
class NLPMeeting(Agent):
    """Agent used for demo for NLP Meeting"""

    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        # shared data
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

        self.spy = location is None
        self.answerer_count = np.zeros(n_players - 1, dtype=int)
        self.round = 0
        self.est_n_rounds_remaining = n_rounds
        self.last_questioner = -1
        self.most_voted_player = random.randint(1, n_players - 1)
        self.question_bank = QUESTIONS.copy()

        # spy data
        self.avg_loc_score = {loc: 0.0 for loc in Location}

        # nonspy data
        self.avg_spy_score = np.zeros(n_players - 1, dtype=float)

        # Generate embeddings of the question/answer bank
        import os
        self.embedding = GeminiEmbedding()
        csv_path = os.path.join(os.path.dirname(__file__), "all_question_bank.csv")
        self.question_data = pd.read_csv(csv_path)
        # Create embeddings for all questions
        
        self.question_data["question_embedding"] = self.question_data["question"].apply(
            lambda x: asyncio.get_event_loop().run_until_complete(self.embedding.get_embeddings(x))
        )
        self.question_data["answer_embedding"] = self.question_data["answer"].apply(
            lambda x: asyncio.get_event_loop().run_until_complete(self.embedding.get_embeddings(x))
        )
        pkl_path = os.path.join(os.path.dirname(__file__), "question_data_with_embeddings.pkl")
        self.question_data.to_pickle(pkl_path)
        # pkl_path = os.path.join(os.path.dirname(__file__), "question_data_with_embeddings.pkl")
        # self.question_data = pd.read_pickle(pkl_path)

        # keeping track of location probability
        self.locs_prob = np.array([0 for loc in Location])

        # keeping track of agent accusation probability
        self.accusation_prob = np.array([0 for i in range(self.n_players - 1)])

    async def ask_question(self) -> tuple[int, str]:
        # select the answerer, excluding the last questioner
        players = list(range(1, self.n_players))
        if self.last_questioner != -1:
            players.remove(self.last_questioner)
        answerer = random.choice(players)

        if self.spy:
            question_list = self.question_data[
                self.question_data["location"] == "General"
            ]
            question = question_list.sample().iloc[0]["question"]
        else:
            question_list = self.question_data[
                self.question_data["location"] == self.location.value
            ]
            question = question_list.sample().iloc[0]["question"]

        return answerer, question

    async def _answer_question_spy(self, question: str) -> str:

        # Get embedding for the current question
        question_embedding = await self.nlp.get_embeddings(question)

        # Calculate euclidean distances between question embedding and all stored embeddings
        distances = self.question_data["question_embedding"].apply(
            lambda x: np.linalg.norm(x - question_embedding)
        )

        # Get the answer from the row with smallest distance
        answer = self.question_data.iloc[distances.argmin()]["answer"]
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        # Get embedding for the current question
        question_embedding = await self.nlp.get_embeddings(question)

        # Calculate euclidean distances between question embedding and all stored embeddings
        distances = self.question_data["question_embedding"].apply(
            lambda x: np.linalg.norm(x - question_embedding)
        )

        # Get the answer from the row with smallest distance
        answer = self.question_data.iloc[distances.argmin()]["answer"]
        return answer

    async def answer_question(self, question: str) -> str:
        question = question[:200]  # limit question length
        if self.spy:
            return await self._answer_question_spy(question)
        else:
            return await self._answer_question_nonspy(question)

    @staticmethod
    def _get_locs_from_str(loc_str: str) -> list[Location]:
        output = []
        for loc in Location:
            if loc.value.lower() in loc_str.lower().strip():
                output.append(loc)
        return output

    @staticmethod
    def _get_float_from_str(float_str: str) -> float | None:
        try:
            return float(float_str)
        except:
            return None

    # Compare embeddings for responses with embeddings for locations
    async def _analyze_response_spy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"Do you see animals here?\" Answer: \"Yes, turtles and fish are common.\""),
            (LLMRole.ASSISTANT, "Turtle Pond, Lake Austin"),
            (LLMRole.USER, f"Question: \"Do people take photos here?\" Answer: \"Yes\""),
            (LLMRole.ASSISTANT, "Blanton Museum of Art, UT Tower, Littlefield Fountain, Turtle Pond, Texas State Capitol, Mt. Bonnell & Mayfield Park, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"Is the architecture grande?\" Answer: \"Yes, it is impressive and historic.\""),
            (LLMRole.ASSISTANT, "Texas State Capitol, UT Tower, Littlefield Fountain, Texas Memorial Museum"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 72, 0.25)
        locs = self._get_locs_from_str(reponse)
        for loc in Location:
            if loc in locs:
                self.avg_loc_score[loc] += 1 / len(locs)
            else:
                self.avg_loc_score[loc] -= 1 / (len(Location) - len(locs))

    async def _analyze_response_nonspy(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.SYSTEM, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Keep in mind people giving vauge answers are more likely to be the spy. Be somewhat aggressive in your guesses here. We wanna catch the spy quickly! The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.ASSISTANT, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 10, 0.25)
        prob = self._get_float_from_str(reponse)
        if prob is not None:
            # Averaging spy scores per player
            if self.answerer_count[answerer - 1] == 1:
                self.avg_spy_score[answerer - 1] += prob
            else:
                # Recalculate weighted average
                current_count = self.answerer_count[answerer - 1]
                self.avg_spy_score[answerer - 1] = (
                    self.avg_spy_score[answerer - 1] * (current_count - 1) + prob
                ) / current_count

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        self.last_questioner = questioner
        if answerer == 0:
            return
        question = question[:200]  # limit question length
        answer = answer[:200]  # limit answer length
        self.answerer_count[answerer - 1] += 1
        if self.spy:
            await self._analyze_response_spy(questioner, question, answerer, answer)
        else:
            await self._analyze_response_nonspy(questioner, question, answerer, answer)

    async def guess_location(self) -> Location | None:
        # if the top location score is 0.5 higher than the second highest, guess that location
        loc_scores = list(self.avg_loc_score.items())
        loc_scores.sort(key=lambda x: x[1], reverse=True)
        if loc_scores[0][1] - loc_scores[1][1] > GUESS_THRESHOLD:
            return loc_scores[0][0]
        return None

    async def accuse_player(self) -> int | None:
        if self.spy:
            if self.most_voted_player == 0:
                return random.randint(1, self.n_players - 1)
            return self.most_voted_player
        else:
            if (
                self.avg_spy_score[np.argmax(self.avg_spy_score)] >= 0.5
                and random.random() < ACCUSE_PROB
            ):
                score = self.avg_spy_score + ACCUSE_NOISE * np.random.normal(
                    0, 1, size=(self.n_players - 1)
                )
                player = int(np.argmax(self.avg_spy_score)) + 1
                return player
            return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        self.round += 1
        percent_voted = sum(vote is not None for vote in votes) / len(votes)
        self.est_n_rounds_remaining = int(
            (self.n_rounds - self.round) * (1 - percent_voted * 2)
        )
        c = Counter(votes)
        del c[None]
        if len(c) > 0:
            self.most_voted_player = c.most_common(1)[0][0]  # type: ignore
