import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd  # type: ignore

from agent import *
from data import *
from game import *
from nlp import *
from submission import *
from util import *

# This file is for internal use only to test the code ####################################



class TestGame(unittest.IsolatedAsyncioTestCase):
    N_SPIES = 2

    @patch("game.AGENT_REGISTRY", {f"Agent{i}": MyAgent for i in range(10)})
    @patch("game.random.choice", lambda x: Location.BEACH)
    @patch("game.random.randint", lambda a, b: 0)
    async def asyncSetUp(self):
        self.player_names = [f"Agent{i}" for i in range(10)]
        self.game = Game(NLP(), self.player_names, n_rounds=20, n_spies=self.N_SPIES)

    async def test_initialization(self):
        self.assertEqual(self.game.n_players, 10)
        self.assertEqual(self.game.player_names, self.player_names)
        self.assertEqual(self.game.n_rounds, 20)
        self.assertEqual(self.game.location, Location.BEACH)
        self.assertEqual(sorted(self.game.spies), [0, 1])
        self.assertEqual(self.game.questioner, 0)
        self.assertEqual(len(self.game.players), 10)
        self.assertEqual(len(self.game.player_nlps), 10)
        self.assertIsInstance(self.game.player_nlps[0], TokenCounterWrapper)
        self.assertEqual(self.game.game_state, GameState.RUNNING)

    async def test_scoring(self):
        self.game.players[0].guess_location = AsyncMock()
        self.game.players[0].guess_location.return_value = Location.BEACH
        await self.game.play_()
        self.assertEqual(self.game.game_state, GameState.SPY_GUESSED_RIGHT)
        target_scores = pd.Series(index=self.player_names, data=[0.0] * 10)
        for i in self.game.spies:
            target_scores[f"Agent{i}"] = 4.0
        self.assertTrue(self.game.get_scores().equals(target_scores))

    async def test_no_one_indicted(self):
        await self.game.play_()
        self.assertEqual(self.game.game_state, GameState.NO_ONE_INDICTED)

    async def test_add_pov(self):
        for i in range(3):
            for j in range(3):
                assert self.game.add_pov(self.game.reverse_pov(j, pov=i), pov=i) == j
                assert self.game.reverse_pov(self.game.add_pov(j, pov=i), pov=i) == j

    @patch("game.AGENT_REGISTRY", {f"Agent{i}": MyAgent for i in range(10)})
    async def test_stress(self):
        # run 1000 games in parallel
        games = [Game(NLP(), self.player_names, n_rounds=20) for _ in range(10)]
        await asyncio.gather(*[game.play_() for game in games])
        for game in games:
            str(game)
            game.get_scores()
            game.get_percent_right_votes()
            self.assertGreater(len(game.rounds), 0)
            game.get_conversation()


class TestRound(unittest.IsolatedAsyncioTestCase):
    @patch("game.AGENT_REGISTRY", {f"Agent{i}": MyAgent for i in range(3)})
    @patch("game.random.choice", lambda x: Location.BEACH)
    @patch("game.random.randint", lambda a, b: 0)
    async def asyncSetUp(self):
        # Setup mock players
        self.nlp = MagicMock(spec=NLP)  # Mock NLP object
        self.player_names = ["Agent0", "Agent1", "Agent2"]
        self.game = Game(self.nlp, self.player_names, n_rounds=1)
        self.round = Round(self.game)

        # Set up player mocks
        for i, player in enumerate(self.game.players):
            player.ask_question = AsyncMock()
            player.answer_question = AsyncMock()
            player.guess_location = AsyncMock()
            player.accuse_player = AsyncMock()
            player.analyze_response = AsyncMock()
            player.analyze_voting = AsyncMock()

    async def test_basic_round(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        p[0].guess_location.return_value = None
        p[0].accuse_player.return_value = None
        p[1].accuse_player.return_value = None
        p[2].accuse_player.return_value = None

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.RUNNING)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, None)
        self.assertEqual(self.round.player_votes, [None] * 3)
        self.assertEqual(self.round.indicted, None)

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_awaited_once()
            p[i].analyze_voting.assert_awaited_once_with([None] * 3)

    async def test_spy_guess_right(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        for i in self.game.spies:
            p[i].guess_location.return_value = Location.BEACH
            p[i].accuse_player.return_value = None
        # TODO we might need to adjust the indices for non spies below
        p[1].accuse_player.return_value = add_pov(0, 1)
        p[2].accuse_player.return_value = add_pov(0, 2)

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.SPY_GUESSED_RIGHT)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, Location.BEACH)
        self.assertFalse(hasattr(self.round, "player_votes"))
        self.assertFalse(hasattr(self.round, "indicted"))

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_not_called()
            p[i].analyze_voting.assert_not_called()

    async def test_spy_guess_wrong(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        p[0].guess_location.return_value = Location.AIRPLANE
        p[0].accuse_player.return_value = None
        p[1].accuse_player.return_value = add_pov(0, 1)
        p[2].accuse_player.return_value = add_pov(0, 2)

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.SPY_GUESSED_WRONG)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, Location.AIRPLANE)
        self.assertFalse(hasattr(self.round, "player_votes"))
        self.assertFalse(hasattr(self.round, "indicted"))

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_not_called()
            p[i].analyze_voting.assert_not_called()

    async def test_vote_tie(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        p[0].guess_location.return_value = None
        p[0].accuse_player.return_value = None
        p[1].accuse_player.return_value = add_pov(0, 1)
        p[2].accuse_player.return_value = add_pov(1, 2)

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.RUNNING)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, None)
        self.assertEqual(self.round.player_votes, [None, 0, 1])
        self.assertEqual(self.round.indicted, None)

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_awaited_once()
            votes_pov = [None, None, None]
            votes_pov[add_pov(1, i)] = add_pov(0, i)
            votes_pov[add_pov(2, i)] = add_pov(1, i)
            p[i].analyze_voting.assert_awaited_once_with(votes_pov)

    async def test_indict_spy(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        p[0].guess_location.return_value = None
        p[0].accuse_player.return_value = add_pov(1, 0)
        p[1].accuse_player.return_value = add_pov(0, 1)
        p[2].accuse_player.return_value = add_pov(0, 2)

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.SPY_INDICTED)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, None)
        self.assertEqual(self.round.player_votes, [1, 0, 0])
        # assert any spy wins
        self.assertIn(self.round.indicted, self.game.spies)

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_awaited_once()
            p[i].analyze_voting.assert_not_called()

    async def test_indict_non_spy(self):
        add_pov = self.game.add_pov
        reverse_pov = self.game.reverse_pov
        p = self.game.players

        p[0].ask_question.return_value = (1, "Question0")
        answerer0 = reverse_pov(1, pov=0)
        p[answerer0].answer_question.return_value = "Answer0"
        p[0].guess_location.return_value = None
        p[0].accuse_player.return_value = add_pov(2, 0)
        p[1].accuse_player.return_value = add_pov(2, 1)
        p[2].accuse_player.return_value = add_pov(0, 2)

        await self.round.play()

        self.assertEqual(self.game.game_state, GameState.NON_SPY_INDICTED)
        self.assertEqual(self.round.questioner, 0)
        self.assertEqual(self.round.question, "Question0")
        self.assertEqual(self.round.answerer, answerer0)
        self.assertEqual(self.round.answer, "Answer0")
        self.assertEqual(self.round.spy_guess, None)
        self.assertEqual(self.round.player_votes, [2, 2, 0])
        self.assertEqual(self.round.indicted, 2)

        p[0].ask_question.assert_awaited_once()
        p[1].ask_question.assert_not_called()
        p[2].ask_question.assert_not_called()
        for i in range(3):
            if i == answerer0:
                p[i].answer_question.assert_awaited_once_with("Question0")
            else:
                p[i].answer_question.assert_not_called()
        p[0].guess_location.assert_awaited_once()
        p[1].guess_location.assert_not_called()
        p[2].guess_location.assert_not_called()
        for i in range(3):
            p[i].analyze_response.assert_awaited_once_with(
                add_pov(0, i), "Question0", add_pov(answerer0, i), "Answer0"
            )
            p[i].accuse_player.assert_awaited_once()
            p[i].analyze_voting.assert_not_called()


class TestUtil(unittest.TestCase):
    def test_redact(self):
        text = "The location is the beach"
        redacted_text = redact(text, Location.BEACH)
        self.assertEqual(redacted_text, "The location is the <REDACTED>")

    def test_count_votes(self):
        votes = [0, 1, 1, 0, None]
        self.assertEqual(count_votes(votes, len(votes)), None)

        votes = [0, 1, 1, 0, 0]
        self.assertEqual(count_votes(votes, len(votes)), 0)

        votes = [0, 1, 1, 0, 0, 1]
        self.assertEqual(count_votes(votes, len(votes)), None)

        votes = [0, None, None]
        self.assertEqual(count_votes(votes, len(votes)), None)


class TestNLP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = DummyLLM()
        self.embedding = DummyEmbedding()
        self.nlp = NLP(llm=self.llm, embedding=self.embedding)
        self.counter = TokenCounterWrapper(self.nlp, token_limit=50)
        self.proxy = NLPProxy(self.counter)

    async def test_counter(self):
        self.llm.prompt = AsyncMock()
        self.llm.prompt.return_value = "".join(["test"] * 100)
        self.embedding.get_embeddings = AsyncMock()
        self.embedding.get_embeddings.return_value = np.ones(768)

        output = await self.proxy.prompt_llm("How are you?")
        self.assertEqual(output, "".join(["test"] * 100))

        # Test that limit has been reached
        output = await self.proxy.prompt_llm("Hi")
        self.assertEqual(output, "")
        emb = await self.proxy.get_embeddings("Hello there")
        self.assertTrue(all(emb == np.zeros(768)))

        # Reset
        self.counter.reset_token_counter()
        output = await self.proxy.prompt_llm("Hi")
        self.assertEqual(output, "".join(["test"] * 100))
        emb = await self.proxy.get_embeddings("Hello there")
        self.assertTrue(all(emb == np.zeros(768)))

        # Reset
        self.counter.reset_token_counter()
        emb = await self.proxy.get_embeddings("Hello there")
        self.assertTrue(all(emb == np.ones(768)))
        output = await self.proxy.prompt_llm("Hi")
        self.assertEqual(output, "".join(["test"] * 100))

    async def test_proxy(self):
        self.assertFalse(hasattr(self.proxy, "nlp"))
        self.assertFalse(hasattr(self.proxy, "reset_token_counter"))


if __name__ == "__main__":
    unittest.main()
