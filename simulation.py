import asyncio
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from glob import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

from agent import *
from game import *
from nlp import *
from util import *

# This file is for internal use to run simulations ####################################
# Use main.py instead #################################################################


@dataclass
class Simulation:
    """A class to run multiple games concurrently and save/analyze the results"""

    nlp: NLP
    agent_names: list[str] | None = None
    gave_save_dir: str = "games/simulation0"
    team_size: int = 3
    n_rounds: int = 20

    def __post_init__(self):
        if self.agent_names is None:
            self.agent_names = list(AGENT_REGISTRY.keys())
        assert self.team_size >= 3, "team size must be at least 3"
        msg = f"the number of agents ({len(self.agent_names)}) must be at least the team size ({self.team_size})"
        assert len(self.agent_names) >= self.team_size, msg
        self.games: list[Game] = []
        self.load_games()

    def validate_agents(self):
        for name in set(self.agent_names):
            try:
                AGENT_REGISTRY[name].validate()
            except Exception as e:
                print(f"Agent {name} failed validation: {e}")
                raise e

    def run(self, n_games: int = 1):
        """Run multiple games in parallel and adds the results to self.games
        Args:
            n_games (int, optional): number of games to run
            agent_names (list[str] | None, optional): names of agent classes to use
        """
        assert self.agent_names is not None
        # Randomly sample agent classes and the spy to play in each game
        sampled_agent_names_and_spies = sample_agents(
            self.agent_names, self.team_size, n_games
        )

        # Set up progress bar
        tqdm_bar = tqdm(
            total=n_games * self.n_rounds,
            desc="Running Simulation, Rounds",
            colour="green",
        )

        # Run games concurrently
        games = [
            Game(self.nlp, agents, self.n_rounds, spy_id)
            for agents, spy_id in sampled_agent_names_and_spies
        ]
        for game in games:
            game.tqdm_bar = tqdm_bar

        event_loop = asyncio.get_event_loop()
        event_loop.run_until_complete(asyncio.gather(*[game.play_() for game in games]))

        for game in games:
            game.tqdm_bar = None
        tqdm_bar.close()

        self.games.extend(games)

    def pickle_games(self):
        """Saves all games to self.gave_save_dir as pickle files"""
        assert len(self.games) > 0, "must call run() or load_games() first"
        os.makedirs(self.gave_save_dir, exist_ok=True)

        for i, game in enumerate(
            tqdm(self.games, desc="Pickling games", colour="green")
        ):
            path = f"{self.gave_save_dir}/game_{i}.pkl"
            with open(path, "wb") as f:
                pickle.dump(game, f)

    def load_games(self):
        """Loads all games from self.gave_save_dir"""
        games = []

        for path in tqdm(
            glob(f"{self.gave_save_dir}/*.pkl"), desc="Loading games", colour="green"
        ):
            with open(path, "rb") as f:
                games.append(pickle.load(f))

        self.games.extend(games)

    def get_scores(self) -> pd.DataFrame:
        """Get the scores of all agents in all games
        Returns:
            pd.DataFrame: Pandas DataFrame with the scores
                columns: agent names
                index: game number
                values: score or np.nan if the agent was not in the game
        """
        assert len(self.games) > 0, "must call run() or load_games() first"
        df = pd.DataFrame(
            index=range(len(self.games)), columns=list(set(self.agent_names))
        )

        for i, game in enumerate(self.games):
            game_scores = game.get_scores()
            assert game_scores.index.isin(df.columns).all()
            df.loc[i] = game_scores

        return df

    def get_percent_right_votes(self) -> pd.DataFrame:
        """Get the percentage of right votes of all agents in all games
        Returns:
            pd.DataFrame: Pandas DataFrame with the percentage of right votes
                columns: agent names
                index: game number
                values: percentage of right votes or np.nan if the agent was not in the game
        """
        assert len(self.games) > 0, "must call run() or load_games() first"
        df = pd.DataFrame(
            index=range(len(self.games)), columns=list(set(self.agent_names))
        )

        for i, game in enumerate(self.games):
            game_scores = game.get_percent_right_votes()
            assert game_scores.index.isin(df.columns).all()
            df.loc[i] = game_scores

        return df

    def get_game_endtype_freq(self) -> pd.Series:
        """Get the frequency of game end types in all games
        Returns:
            pd.Series: Pandas Series with the frequency of each game end type
                index: game end type
                values: frequency
        """
        c = Counter(g.game_state for g in self.games)
        return pd.Series(c)

    def get_game_duration_freq(self) -> pd.Series:
        """Get the frequency of game durations in all games
        Returns:
            pd.Series: Pandas Series with the frequency of each game duration
                index: game duration
                values: frequency
        """
        c = Counter(len(g.rounds) for g in self.games)
        return pd.Series(c)

    def _get_animation(
        self, duration: int = 10, fps: int = 30
    ) -> animation.FuncAnimation:
        df = self.get_scores()

        for col in df.columns:
            # Use expanding mean times index instead of cumsum to correctly handle NaNs
            df[col] = df[col].expanding().mean() * (df.index + 1)
            df[col] = df[col].fillna(0)

        plt.rcParams["toolbar"] = "None"

        fig, ax = plt.subplots()
        lines = [ax.plot([], [], lw=2, label=col)[0] for col in df.columns]
        ax.set_xlabel("Game")
        ax.set_ylabel("Cumulative Score")
        ax.set_title("Results")
        ax.legend(loc="upper left")

        def animate(animation_i):
            percent_complete = animation_i / (duration * fps)

            # adjust growth rate here to be non-linear if desired
            i = int(min(percent_complete, 1) * len(df))

            x = df.index[:i]
            for j, line in enumerate(lines):
                y = df[df.columns[j]][:i]
                line.set_data(x, y)

            # scale graph axes
            if i > 0:
                ax.set_xlim(0, max(1, x.max()))
                ax.set_ylim(0, max(1, df.values[:i].max()))

            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=duration * fps + 1,
            interval=1000 / fps,
            repeat=False,
        )

        return ani

    def visualize_scores(self, duration: int = 10, fps: int = 30):
        """Plays visualization of the scores of all agents over time"""
        ani = self._get_animation(duration, fps)

        plt.show()

    def save_visualization(self, filepath, duration: int = 10, fps: int = 30):
        """Saves visualization of the scores of all agents over time to a file"""
        ani = self._get_animation(duration, fps)

        ani.save(filepath, writer="ffmpeg")
