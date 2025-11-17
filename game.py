import asyncio
import random
from enum import Enum
from functools import lru_cache
from itertools import chain

import pandas as pd  # type: ignore
import pygame
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from agent import AGENT_REGISTRY, Agent
from data import *
from nlp import *
from util import *
from visualizations import Visualization

# This file is for internal use to run the game ####################################
# Use main.py instead ##############################################################


# Used to describe how the game ended or if it is ongoing
class GameState(Enum):
    RUNNING = "running"
    SPY1_GUESSED_RIGHT = "spy1 guessed right"
    SPY2_GUESSED_RIGHT = "spy2 guessed right"
    SPY1_GUESSED_WRONG = "spy1 guessed wrong"
    SPY2_GUESSED_WRONG = "spy2 guessed wrong"
    SPY1_INDICTED = "spy1 indicted"
    SPY2_INDICTED = "spy2 indicted"
    NON_SPY_INDICTED = "non-spy indicted"
    NO_ONE_INDICTED = "no one indicted"


class Game:
    """Used to run a game of Spyfall
    Consists of multiple rounds
    """

    n_players: int
    player_names: list[str]
    n_rounds: int

    location: Location
    spies: list[int]  # list of spy player indices
    questioner: int  # current questioner
    last_questioner: int = -1  # last questioner
    players: list[Agent]
    player_nlps: list[TokenCounterWrapper]  # each keeps track of tokens per round
    rounds: list["Round"]
    game_state: GameState
    indicted_spy: int | None = None  # which spy was indicted (if any)
    guessing_spy: int | None = None  # which spy made the guess (if any)

    # can be optionally set to visualize how many rounds have been played
    tqdm_bar: tqdm | None = None

    spy1_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY1_GUESSED_RIGHT: 4,
        GameState.SPY2_GUESSED_RIGHT: 0,
        GameState.SPY1_GUESSED_WRONG: 0,
        GameState.SPY2_GUESSED_WRONG: 1,
        GameState.SPY1_INDICTED: 0,
        GameState.SPY2_INDICTED: 4,
        GameState.NON_SPY_INDICTED: 3,
        GameState.NO_ONE_INDICTED: 2,
    }
    spy2_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY1_GUESSED_RIGHT: 0,
        GameState.SPY2_GUESSED_RIGHT: 4,
        GameState.SPY1_GUESSED_WRONG: 1,
        GameState.SPY2_GUESSED_WRONG: 0,
        GameState.SPY1_INDICTED: 4,
        GameState.SPY2_INDICTED: 0,
        GameState.NON_SPY_INDICTED: 3,
        GameState.NO_ONE_INDICTED: 2,
    }
    nonspy_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY1_GUESSED_RIGHT: 0,
        GameState.SPY2_GUESSED_RIGHT: 0,
        GameState.SPY1_GUESSED_WRONG: 1,
        GameState.SPY2_GUESSED_WRONG: 1,
        GameState.SPY1_INDICTED: 1,
        GameState.SPY2_INDICTED: 1,
        GameState.NON_SPY_INDICTED: 0,
        GameState.NO_ONE_INDICTED: 0,
    }

    def __init__(
        self,
        nlp: NLP,
        player_names: list[str] | None = None,
        n_rounds: int = 20,
        spy_ids: list[int] | None = None,
    ):
        # init game
        if player_names is None:
            player_names = list(AGENT_REGISTRY.keys())
        n_players = self.n_players = len(player_names)
        assert n_players >= 3, "need at least 3 players"
        self.player_names = player_names
        self.n_rounds = n_rounds

        self.location = random.choice(list(Location))
        if spy_ids is not None:
            self.spies = spy_ids
        else:
            # Randomly assign 2 players as spies
            self.spies = random.sample(range(n_players), 2)
        self.questioner = random.randint(0, n_players - 1)
        self.players = []
        self.player_nlps = []
        self.rounds: list[Round] = []
        self.game_state = GameState.RUNNING

        for i, player_class_name in enumerate(player_names):
            player_class = AGENT_REGISTRY[player_class_name]
            player_nlp = TokenCounterWrapper(nlp, player_name=player_class_name)
            given_location = self.location if i not in self.spies else None
            player_instance = player_class(
                given_location, n_players, n_rounds, nlp=NLPProxy(player_nlp)
            )
            self.players.append(player_instance)
            self.player_nlps.append(player_nlp)

        # povs maps global player index to local player index
        self._povs = [list(range(1, n_players)) for _ in range(n_players)]
        for i, pov in enumerate(self._povs):
            random.shuffle(pov)
            pov.insert(i, 0)  # global index i always maps to local index 0

        # r_povs maps local player index to global player index
        self._r_povs = [[0] * (n_players) for _ in range(n_players)]
        for i in range(n_players):
            for player, player_w_pov in enumerate(self._povs[i]):
                self._r_povs[i][player_w_pov] = player

    def add_pov(self, player: int, pov: int):
        """adds a point of view to a player index"""
        return self._povs[pov][player]

    def reverse_pov(self, player: int, pov: int):
        """Remove a point of view from a player index"""
        return self._r_povs[pov][player]

    def play(self):
        """runs the game"""
        asyncio.get_event_loop().run_until_complete(self.play_())

    async def play_(self):
        tqdm_bar = (
            self.tqdm_bar
            if self.tqdm_bar
            else tqdm(total=self.n_rounds, desc="Running Game, Rounds", colour="green")
        )

        for _ in range(self.n_rounds):
            round = Round(self)
            await round.play()
            tqdm_bar.update(1)
            self.rounds.append(round)
            if self.game_state != GameState.RUNNING:
                break
        else:
            self.game_state = GameState.NO_ONE_INDICTED
        tqdm_bar.update(self.n_rounds - len(self.rounds))

    def get_scores(self) -> pd.Series:
        """Gets the scores of all players as a pandas series

        Returns:
            pd.Series: Pandas Series with the scores
                index: player names
                values: score
        """
        scores_list = [self.nonspy_scoring[self.game_state]] * self.n_players
        scores_list[self.spies[0]] = self.spy1_scoring[self.game_state]
        scores_list[self.spies[1]] = self.spy2_scoring[self.game_state]

        scores = pd.Series(data=scores_list, index=self.player_names)
        scores = scores.groupby(scores.index).mean()
        return scores

    def get_percent_right_votes(self) -> pd.Series:
        """Gets the percent of right votes for each player as a pandas series

        Returns:
            pd.Series: Pandas Series with the percent of right votes
                index: player names
                values: percent of right votes
        """
        votes = np.array(
            [
                round.player_votes
                for round in self.rounds
                if hasattr(round, "player_votes")
            ]
        )

        if len(votes) == 0:
            return pd.Series(index=list(set(self.player_names)))

        # Calculate percent right votes for each spy
        percent_right_votes = np.zeros(self.n_players)
        for i in self.spies:
            percent_right_votes += np.mean(votes == i, axis=0)
        series = pd.Series(data=percent_right_votes, index=self.player_names)
        series = series.groupby(series.index).mean()
        return series

    @lru_cache
    def get_conversation(self) -> pd.DataFrame:
        """Gets the conversation as a pandas dataframe

        Returns:
            pd.DataFrame: Pandas DataFrame with the conversation
                columns: player id, message
        """
        conv_list = list(chain(*[round.get_conversation() for round in self.rounds]))

        if self.game_state == GameState.NO_ONE_INDICTED:
            no_one_indicted_msg = random.choice(NO_ONE_INDICTED_RESPONSE)
            player = random.choice(list(set(range(self.n_players)) - set(self.spies)))
            conv_list.append((player, no_one_indicted_msg))

            # both spies reveal themselves
            spy1_reveal_msg = random.choice(SPY_REVEAL)
            conv_list.append((self.spies[0], spy1_reveal_msg))
            spy2_reveal_msg = random.choice(SPY_REVEAL)
            conv_list.append((self.spies[1], spy2_reveal_msg))

        df = pd.DataFrame(conv_list, columns=["player", "message"])
        df["player_name"] = df["player"].apply(lambda x: self.player_names[x])
        # set column order to player, player_name, message
        df = df[["player", "player_name", "message"]]
        return df

    def save_conversation(self, path: str):
        """Saves the conversation to a path
        Converts to 1-indexed player ids"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = self.get_conversation()
        df["player"] += 1
        df.to_csv(path, index=False)

    def pregenerate_audio(self):
        """pre-generates audio for the game"""
        for round in tqdm(
            self.rounds, desc="Pregenerating Audio, Rounds", colour="green"
        ):
            round.pregenerate_audio()

    def render(self):
        assert self.rounds[0].audio, "need to pregenerate audio first"

        """Visualizes the game and plays the audio"""
        # set audio output to system default
        os.environ["SDL_AUDIODRIVER"] = "coreaudio"

        # setup audio
        sr = self.rounds[0].audio[0][2]
        pygame.mixer.pre_init(frequency=sr, channels=1, allowedchanges=0)
        pygame.init()

        # init visualization
        vis = Visualization(self.player_names, self.spies, self.location.value)

        for round in self.rounds:
            round.render(vis)

        # close pygame
        del vis

    def save_audio(self, path: str):
        """saves the audio to a path"""
        self.rounds[0].audio, "need to pregenerate audio"
        sr = self.rounds[0].audio[0][2]
        audio_list = [a for round in self.rounds for _, a, _ in round.audio]
        audio_list = [np.pad(a, (0, sr // 2)) for a in audio_list]
        comb_audio = np.concatenate(audio_list)
        # save audio to path
        sf.write(path, comb_audio, sr)

    def __str__(self):
        return f"Location: {self.location}, Spies: {self.spies}, Ending: {self.game_state}"


class Round:
    """Used to run a round of Spyfall
    Uses a game object to track the current state
    Uses instance variables to log the round's events"""

    questioner: int
    question: str
    answerer: int
    answer: str

    spy_guess: Location | None

    player_votes: list[int | None]
    indicted: int | None

    def __init__(self, game: Game):
        self.game = game

    async def play(self):
        game = self.game
        questioner = self.questioner = game.questioner

        # reset token counter for each player
        for nlp in game.player_nlps:
            nlp.reset_token_counter()

        # ask question
        answerer, question = await game.players[questioner].ask_question()
        assert 1 <= answerer < game.n_players and isinstance(question, str)
        answerer = game.reverse_pov(answerer, pov=questioner)

        # if the questioner selected the last questioner, select a random player
        if game.last_questioner != -1 and answerer == game.last_questioner:
            print(
                f"questioner ({type(game.players[questioner])}) selected last questioner, selecting random player"
            )
            answerer = random.choice(
                [
                    i
                    for i in range(game.n_players)
                    if i != questioner and i != game.last_questioner
                ]
            )

        # answer question
        answer = await game.players[answerer].answer_question(question)
        assert isinstance(answer, str)

        # send question and answer to all players
        futures = []
        for player in range(game.n_players):
            q = game.add_pov(questioner, pov=player)
            a = game.add_pov(answerer, pov=player)
            futures.append(
                game.players[player].analyze_response(q, question, a, answer)
            )
        await asyncio.gather(*futures)

        self.question = question
        self.answer = answer
        game.last_questioner = questioner
        game.questioner = self.answerer = answerer

        # spy voting - handle 2 spies with randomization
        spy_guesses = []
        for spy in game.spies:
            guess = await game.players[spy].guess_location()
            assert guess is None or isinstance(guess, Location)
            spy_guesses.append((spy, guess))
        
        # Randomize order of spy guesses
        random.shuffle(spy_guesses)
        
        # Check guesses in random order
        for spy, guess in spy_guesses:
            if guess == game.location:
                # Determine which spy guessed correctly
                if spy == game.spies[0]:
                    game.game_state = GameState.SPY1_GUESSED_RIGHT
                else:
                    game.game_state = GameState.SPY2_GUESSED_RIGHT
                game.guessing_spy = spy
                self.spy_guess = guess
                return
            elif guess is not None:
                # Determine which spy guessed incorrectly
                if spy == game.spies[0]:
                    game.game_state = GameState.SPY1_GUESSED_WRONG
                else:
                    game.game_state = GameState.SPY2_GUESSED_WRONG
                game.guessing_spy = spy
                self.spy_guess = guess
                return
        
        # No spy guessed
        self.spy_guess = None

        # collect votes
        votes = await asyncio.gather(
            *[player.accuse_player() for player in game.players]
        )
        assert all(1 <= vote < game.n_players for vote in votes if vote is not None)
        for i, vote in enumerate(votes):
            if vote is not None:
                votes[i] = game.reverse_pov(vote, pov=i)

        self.player_votes = votes

        # count votes
        indicted = self.indicted = count_votes(votes, game.n_players)
        if indicted in game.spies:
            # Determine which spy was indicted
            if indicted == game.spies[0]:
                game.game_state = GameState.SPY1_INDICTED
            else:
                game.game_state = GameState.SPY2_INDICTED
            game.indicted_spy = indicted
            return
        elif indicted is not None:
            game.game_state = GameState.NON_SPY_INDICTED
            return

        # send votes to players
        futures = []
        for i in range(game.n_players):
            votes_pov = [None] * game.n_players
            for voter, votee in enumerate(votes):
                if votee is None:
                    continue
                voter = game.add_pov(voter, pov=i)
                votee = game.add_pov(votee, pov=i)
                votes_pov[voter] = votee
            futures.append(game.players[i].analyze_voting(votes_pov))
        await asyncio.gather(*futures)

    @lru_cache
    def get_conversation(self) -> list[tuple[int, str]]:
        """returns the conversation as a list of tuples of player index and their message"""
        game = self.game
        names = self.game.player_names
        output = []

        # question and answer
        output.append((self.questioner, f"{names[self.answerer]}, {self.question}"))
        output.append((self.answerer, self.answer))

        # spy guess
        if self.spy_guess is not None:
            spy = random.choice(game.spies)
            # spy: I am the spy. Was it the {location}?
            msg = random.choice(SPY_REVEAL_AND_GUESS).format(
                location=self.spy_guess.value
            )
            output.append((game.guessing_spy, msg))
            responder = random.choice(list(set(range(game.n_players)) - set(game.spies)))
            if game.game_state in [GameState.SPY1_GUESSED_RIGHT, GameState.SPY2_GUESSED_RIGHT]:
                # random nonspy: yes that is right
                msg = random.choice(SPY_GUESS_RIGHT_RESPONSE)
            else:
                # random nonspy: no, it was the {location}
                msg = random.choice(SPY_GUESS_WRONG_RESPONSE).format(
                    location=game.location.value
                )
            output.append((responder, msg))

        # indictment
        elif self.indicted is not None:
            # one of the accusers: "I think it's player {spy} are you the spy?"
            accuser = random.choice(
                [i for i, x in enumerate(self.player_votes) if x == self.indicted]
            )
            msg = random.choice(ACCUSATION).format(spy=names[self.indicted])
            output.append((accuser, msg))
            if game.game_state in [GameState.SPY1_INDICTED, GameState.SPY2_INDICTED]:
                # spy: I am the spy
                msg = random.choice(SPY_INDICTED_RESPONSE)
                output.append((self.indicted, msg))
            else:
                # indicted: No, I am not the spy
                msg = random.choice(NON_SPY_INDICTED_RESPONSE)
                output.append((self.indicted, msg))
                # spy: I am the spy
                msg = random.choice(SPY_REVEAL)
                output.append((game.spies[0], msg))

        return output

    def pregenerate_audio(self):
        """pre-generates audio for the game"""
        # list of (player, audio, sr)
        self.audio: list[int, np.ndarray, int] = []
        for player, message in self.get_conversation():
            message = message.replace("<out of tokens>", "")
            voice, ps = get_voice_and_ps(self.game.player_names[player])
            audio, sr = text_to_speech(message, voice, ps)
            self.audio.append((player, audio, sr))

    def render(self, vis: Visualization):
        conv = self.get_conversation()
        # render the round
        self.audio, "need to pregenerate audio"
        # game = self.game
        # print(game.window)
        for voice, (player_id, msg) in zip(self.audio, conv):
            vis.render_text(player_id, msg)
            _, audio, _ = voice
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
