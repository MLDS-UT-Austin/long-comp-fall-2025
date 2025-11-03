import asyncio
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from agent_data import *

from agent import Agent, register_agent
from data import Location
from nlp import GeminiEmbedding, LLMRole, NLPProxy

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
            (LLMRole.USER, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.MODEL, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.USER, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"How much would you gamble at this location\" Answer: \"Hopefully not too much\""),
            (LLMRole.MODEL, "Casino"),
            (LLMRole.USER, f"Question: \"Have you been to this location before?\" Answer: \"No\""),
            (LLMRole.MODEL, "Corporate Party, Crusader Army, Embassy, Ocean Liner, Pirate Ship, Polar Station, Space Station, Submarine"),
            (LLMRole.USER, f"Question: \"How far is this location from the nearest public transportation?\" Answer: \"Very very far\""),
            (LLMRole.MODEL, "Airplane, Crusader Army, Ocean Liner, Space Station, Submarine, Pirate Ship, Polar Station"),
            (LLMRole.USER, f"Question: \"{question}\" Answer: \"{answer}\""),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 200, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 40, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.MODEL, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.USER, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. List of only the most reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"How much would you gamble at this location\" Answer: \"Hopefully not too much\""),
            (LLMRole.MODEL, "Casino"),
            (LLMRole.USER, f"Question: \"Have you been to this location before?\" Answer: \"No\""),
            (LLMRole.MODEL, "Corporate Party, Crusader Army, Embassy, Ocean Liner, Pirate Ship, Polar Station, Space Station, Submarine"),
            (LLMRole.USER, f"Question: \"How far is this location from the nearest public transportation?\" Answer: \"Very very far\""),
            (LLMRole.MODEL, "Airplane, Crusader Army, Ocean Liner, Space Station, Submarine, Pirate Ship, Polar Station"),
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
            (LLMRole.USER, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Be a bit passive about your guesses, but keep in mind people giving vauge answers are more likely to be the spy. The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 40, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. You are the spy. Answer as vaguely as possible to reduce suspicion."),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, "Not as often as some other locations."),
            (LLMRole.USER, "What time of day is the busiest here?"),
            (LLMRole.MODEL, "It's hard to say, it depends on the week."),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
        return answer

    async def _answer_question_nonspy(self, question: str) -> str:
        assert self.location
        # fmt: off
        prompt = [
            (LLMRole.USER, f"You are playing a game of spyfall. Answer the question based off the location which is the {self.location.value}. DO NOT REVEAL THE LOCATION IN YOUR ANSWER!"),
            (LLMRole.USER, "How often do you come here?"),
            (LLMRole.MODEL, EXAMPLE_ANSWER[self.location]),
            (LLMRole.USER, question),
        ]
        # fmt: on
        answer = await self.nlp.prompt_llm(prompt, 40, 0.25)
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
            (LLMRole.USER, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"How much would you gamble at this location\" Answer: \"Hopefully not too much\""),
            (LLMRole.MODEL, "Casino"),
            (LLMRole.USER, f"Question: \"Have you been to this location before?\" Answer: \"No\""),
            (LLMRole.MODEL, "Corporate Party, Crusader Army, Embassy, Ocean Liner, Pirate Ship, Polar Station, Space Station, Submarine"),
            (LLMRole.USER, f"Question: \"How far is this location from the nearest public transportation?\" Answer: \"Very very far\""),
            (LLMRole.MODEL, "Airplane, Crusader Army, Ocean Liner, Space Station, Submarine, Pirate Ship, Polar Station"),
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
            (LLMRole.USER, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Keep in mind people giving vauge answers are more likely to be the spy. Be somewhat aggressive in your guesses here. We wanna catch the spy quickly! The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 40, 0.25)
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

        assert os.path.exists(
            "question_data_with_embeddings.pkl"
        ), "Please run generate_embeddings.py to generate question_data_with_embeddings.pkl"
        self.question_data = pd.read_pickle("question_data_with_embeddings.pkl")

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
            (LLMRole.USER, f"You are playing a game of spyfall. List of ALL reasonable locations for each question and answer. Possible locations: {', '.join([i.value for i in Location])}"),
            (LLMRole.USER, f"Question: \"How much would you gamble at this location\" Answer: \"Hopefully not too much\""),
            (LLMRole.MODEL, "Casino"),
            (LLMRole.USER, f"Question: \"Have you been to this location before?\" Answer: \"No\""),
            (LLMRole.MODEL, "Corporate Party, Crusader Army, Embassy, Ocean Liner, Pirate Ship, Polar Station, Space Station, Submarine"),
            (LLMRole.USER, f"Question: \"How far is this location from the nearest public transportation?\" Answer: \"Very very far\""),
            (LLMRole.MODEL, "Airplane, Crusader Army, Ocean Liner, Space Station, Submarine, Pirate Ship, Polar Station"),
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
            (LLMRole.USER, f"You are playing a game of spyfall. Give the probability of the answerer being the spy. Keep in mind people giving vauge answers are more likely to be the spy. Be somewhat aggressive in your guesses here. We wanna catch the spy quickly! The location is the {self.location.value}."),
            (LLMRole.USER, f"Question: \"{QUESTIONS[2]}\", Answer: \"I don't know\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_BAD_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "1.0"),
            (LLMRole.USER, f"Question: \"{QUESTIONS[0]}\", Answer: \"{EXAMPLE_ANSWER[self.location]}\""),
            (LLMRole.MODEL, "0.0"),
            (LLMRole.USER, f"Question: {question}, Answer: {answer}"),
        ]
        # fmt: on
        reponse = await self.nlp.prompt_llm(prompt, 40, 0.25)
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


@register_agent("Pattern Matcher")
class PatternMatcher(Agent):
    """
    Custom Agent for 2-Spy Spyfall Game
    Strategy: Frequency analysis and pattern matching
    Author: Sydney

    Agent that uses word frequency analysis and answer patterns to detect spies.

    Key differences from existing agents:
    1. Uses word frequency analysis instead of LLM or embeddings
    2. Tracks answer length and repetition patterns
    3. Simple rule-based answering (no LLM generation)
    4. Designed for 2-spy games with adaptive suspicion thresholds
    5. Uses answer consistency checking across rounds
    """

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

        self.spy = location is None
        self.round = 0
        self.last_questioner = -1

        # Track player behavior patterns
        self.player_answers = {i: [] for i in range(1, n_players)}  # Store all answers
        self.player_word_freq = {
            i: Counter() for i in range(1, n_players)
        }  # Word frequency
        self.player_avg_length = np.zeros(
            n_players - 1, dtype=float
        )  # Average answer length
        self.answerer_count = np.zeros(n_players - 1, dtype=int)

        # Suspicion tracking (for 2-spy logic)
        self.suspicion_scores = np.zeros(n_players - 1, dtype=float)

        # Spy-specific: track location mentions
        if self.spy:
            self.location_mentions = Counter()
            self.has_guessed = False

        # Answer templates based on location type
        self.location_answers = {
            "indoor": [
                "The interior is well-maintained.",
                "It's climate-controlled, which is nice.",
                "The layout is pretty standard for this type of place.",
            ],
            "outdoor": [
                "The weather affects the experience here.",
                "It's nice when the conditions are right.",
                "Being outside is part of the appeal.",
            ],
            "transport": [
                "You need to get from one place to another.",
                "It's about the journey, really.",
                "Timing is important here.",
            ],
            "service": [
                "People come here for a specific purpose.",
                "The staff are usually helpful.",
                "It serves an important function.",
            ],
            "entertainment": [
                "People come here to enjoy themselves.",
                "It's a popular destination.",
                "The experience varies each time.",
            ],
        }

        # Vague spy answers
        self.spy_answers = [
            "It's pretty typical, I'd say.",
            "Depends on the day, really.",
            "I don't come here that often.",
            "It's similar to other places.",
            "Hard to describe exactly.",
            "It varies quite a bit.",
        ]

        # Question bank
        self.questions = [
            "What brings people here?",
            "How would you describe this place?",
            "What's distinctive about here?",
            "When do you usually come here?",
            "What do you like most about this place?",
            "How long do people typically stay?",
            "What's the atmosphere like?",
            "Who else comes here?",
        ]
        random.shuffle(self.questions)

    async def ask_question(self) -> tuple[int, str]:
        """
        Select who to question and what to ask.
        Strategy: Balance between unexplored players and suspicious ones.
        """
        # Avoid asking the last questioner
        available_players = list(range(1, self.n_players))
        if self.last_questioner != -1 and self.last_questioner in available_players:
            available_players.remove(self.last_questioner)

        if len(available_players) == 0:
            available_players = list(range(1, self.n_players))

        # Target most suspicious player 40% of the time if we have suspicions
        if not self.spy and self.round > 2:
            max_suspicion_idx = int(np.argmax(self.suspicion_scores))
            if self.suspicion_scores[max_suspicion_idx] > 2.0 and random.random() < 0.4:
                answerer = max_suspicion_idx + 1
                if answerer in available_players:
                    question = self.questions[self.round % len(self.questions)]
                    return answerer, question

        # Otherwise, prefer players we haven't asked much
        if len(available_players) > 0:
            weights = 1.0 / (self.answerer_count[np.array(available_players) - 1] + 1)
            weights = weights / weights.sum()
            answerer = int(np.random.choice(available_players, p=weights))
        else:
            answerer = random.randint(1, self.n_players - 1)

        # Cycle through questions
        question = self.questions[self.round % len(self.questions)]
        return answerer, question

    async def answer_question(self, question: str) -> str:
        """
        Answer questions using simple rule-based templates.
        """
        question_lower = question.lower()

        if self.spy:
            # Spies give vague answers
            return random.choice(self.spy_answers)
        else:
            # Non-spies give category-appropriate answers
            return self._get_location_answer(question_lower)

    def _get_location_answer(self, question: str) -> str:
        """Get an answer based on location category."""
        assert self.location is not None

        # Categorize locations
        indoor_locations = [
            "BANK",
            "CASINO",
            "CATHEDRAL",
            "HOSPITAL",
            "HOTEL",
            "RESTAURANT",
            "SCHOOL",
            "SUPERMARKET",
        ]
        outdoor_locations = ["BEACH", "CIRCUS_TENT", "MILITARY_BASE", "POLAR_STATION"]
        transport_locations = [
            "AIRPLANE",
            "OCEAN_LINER",
            "PASSENGER_TRAIN",
            "PIRATE_SHIP",
            "SPACE_STATION",
            "SUBMARINE",
        ]
        service_locations = ["DAY_SPA", "EMBASSY", "POLICE_STATION", "SERVICE_STATION"]
        entertainment_locations = [
            "BROADWAY_THEATER",
            "CORPORATE_PARTY",
            "MOVIE_STUDIO",
        ]

        location_name = self.location.name

        if location_name in indoor_locations:
            return random.choice(self.location_answers["indoor"])
        elif location_name in outdoor_locations:
            return random.choice(self.location_answers["outdoor"])
        elif location_name in transport_locations:
            return random.choice(self.location_answers["transport"])
        elif location_name in service_locations:
            return random.choice(self.location_answers["service"])
        elif location_name in entertainment_locations:
            return random.choice(self.location_answers["entertainment"])
        else:
            return "It serves its purpose well."

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        answer: str,
    ) -> None:
        """
        Analyze responses using word frequency and pattern analysis.
        """
        self.last_questioner = questioner

        if answerer == 0:  # Skip if we answered
            return

        self.answerer_count[answerer - 1] += 1
        answer_lower = answer.lower()

        # Store answer
        self.player_answers[answerer].append(answer_lower)

        # Update word frequency
        words = answer_lower.split()
        self.player_word_freq[answerer].update(words)

        # Update average answer length
        total_length = sum(len(ans) for ans in self.player_answers[answerer])
        self.player_avg_length[answerer - 1] = total_length / len(
            self.player_answers[answerer]
        )

        if self.spy:
            # As spy: look for location clues in answers
            self._analyze_as_spy(answer_lower)
        else:
            # As non-spy: detect suspicious patterns
            self._analyze_as_nonspy(answerer, answer_lower)

    def _analyze_as_spy(self, answer: str):
        """Look for location mentions to guess where we are."""
        # Track common words that might indicate location type
        location_hints = {
            "indoor": ["inside", "building", "room", "floor", "ceiling"],
            "outdoor": ["outside", "weather", "sky", "sun", "air"],
            "water": ["water", "ocean", "sea", "boat", "ship"],
            "transport": ["travel", "journey", "destination", "arrive"],
            "service": ["help", "service", "staff", "customer"],
        }

        for category, keywords in location_hints.items():
            for keyword in keywords:
                if keyword in answer:
                    self.location_mentions[category] += 1

    def _analyze_as_nonspy(self, answerer: int, answer: str):
        """Detect spies using pattern analysis."""
        # Vague words that spies might use
        vague_indicators = [
            "depends",
            "varies",
            "sometimes",
            "maybe",
            "typical",
            "similar",
            "usually",
            "often",
            "generally",
            "kind of",
        ]

        # Specific words that non-spies would use
        specific_indicators = [
            "always",
            "never",
            "exactly",
            "specifically",
            "definitely",
            "particular",
            "certain",
        ]

        vague_count = sum(1 for word in vague_indicators if word in answer)
        specific_count = sum(1 for word in specific_indicators if word in answer)

        # Short answers are suspicious
        length_penalty = 0
        if len(answer) < 30:
            length_penalty = 1.0
        elif len(answer) < 50:
            length_penalty = 0.5

        # Calculate suspicion
        suspicion_delta = vague_count - specific_count + length_penalty
        self.suspicion_scores[answerer - 1] += suspicion_delta

        # Check for repetitive answers (spies might reuse templates)
        if len(self.player_answers[answerer]) >= 2:
            last_two = self.player_answers[answerer][-2:]
            common_words = set(last_two[0].split()) & set(last_two[1].split())
            if len(common_words) > 3:  # Too much overlap
                self.suspicion_scores[answerer - 1] += 0.5

    async def guess_location(self) -> Location | None:
        """
        Spy tries to guess location based on accumulated hints.
        Conservative strategy: only guess with strong evidence.
        """
        if not self.spy or self.has_guessed:
            return None

        # Need at least 3 rounds of data
        if self.round < 3:
            return None

        # Only guess if we have strong hints
        if len(self.location_mentions) == 0:
            return None

        most_common_category = self.location_mentions.most_common(1)[0]

        # Need at least 4 mentions to be confident
        if most_common_category[1] < 4:
            return None

        # Map categories to likely locations (simplified)
        category_to_locations = {
            "water": [Location.OCEAN_LINER, Location.PIRATE_SHIP, Location.SUBMARINE],
            "transport": [
                Location.AIRPLANE,
                Location.PASSENGER_TRAIN,
                Location.SPACE_STATION,
            ],
            "service": [Location.HOSPITAL, Location.POLICE_STATION, Location.DAY_SPA],
        }

        category = most_common_category[0]
        if category in category_to_locations:
            self.has_guessed = True
            return random.choice(category_to_locations[category])

        return None

    async def accuse_player(self) -> int | None:
        """
        Accuse a player of being a spy.
        2-spy aware: uses adaptive threshold based on game progress.
        """
        if self.spy:
            # As spy: vote randomly to blend in
            if random.random() < 0.3:  # Don't always vote
                return random.randint(1, self.n_players - 1)
            return None

        # As non-spy: accuse based on suspicion scores
        # Need at least 2 rounds of data
        if self.round < 2:
            return None

        # Find most suspicious player
        max_suspicion_idx = int(np.argmax(self.suspicion_scores))
        max_suspicion = self.suspicion_scores[max_suspicion_idx]

        # Adaptive threshold: lower in late game
        threshold = 3.0 if self.round < self.n_rounds // 2 else 2.0

        # Only accuse if suspicion exceeds threshold
        if max_suspicion > threshold and self.answerer_count[max_suspicion_idx] >= 2:
            # 40% chance to actually vote (conservative)
            if random.random() < 0.4:
                return max_suspicion_idx + 1

        return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        """
        Analyze voting patterns to refine suspicions.
        """
        self.round += 1

        # Count votes
        vote_counts = Counter([v for v in votes if v is not None])

        if not self.spy and len(vote_counts) > 0:
            # If many people vote for someone, they might be onto something
            for player, count in vote_counts.items():
                if player > 0 and count >= 2:  # At least 2 votes
                    # Increase suspicion slightly (crowd wisdom)
                    self.suspicion_scores[player - 1] += 0.3 * count
