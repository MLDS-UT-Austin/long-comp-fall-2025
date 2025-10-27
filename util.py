import asyncio
import functools
import importlib
import os
import random
import re
import sys
import time
from collections import Counter
from functools import wraps
from glob import glob
from os.path import dirname

import gtts  # type: ignore
import numpy as np
import parselmouth
from gtts import gTTS  # type: ignore
from parselmouth.praat import call
from together.error import RateLimitError

from data import *

# Feel free to use redact() in your agent:


def redact(text: str, location: Location, redacted_text: str = "<REDACTED>") -> str:
    """Can optionally by used by agents to redact text based on the location
    This can be useful to prevent the LLM from giving away the location
    Note: this is not called in game.py
    Args:
        text (str): text to redact
        location (Location): the location to redact
        redacted_text (str, optional): what to replace the redacted text with
    """
    for word in redaction_dict[location]:
        text = re.sub(rf"{word}", redacted_text, text, flags=re.IGNORECASE)
    return text


# Everything below is for internal use ###############################################


def count_votes(votes: list[int | None], n_players: int) -> int | None:
    """Used in game.py to count votes and determine the majority
    Args:
        votes (list[int  |  None]): the player that each player voted for, or None if they abstained
        n_players (int): the number of players in the game
    """
    counter = Counter(votes)
    if counter[None] >= n_players / 2:
        return None  # half or more abstained
    else:
        del counter[None]
        top2 = counter.most_common(2)  # ((player, count), (player, count))
        if len(top2) == 1 or top2[0][1] > top2[1][1]:
            return top2[0][0]
        else:
            return None  # tie


def sample_agents(
    agent_names: list[str],
    team_size: int,
    n_games: int,
    n_spies: int = 1,
    verbose: bool = False,
    max_same_agent: int = 2,
) -> list[tuple[list[str], int]]:
    """Sample agents for a game, ensuring fairness for both game count and spy count

    Args:
        agent_names (list[str]): list of agent names
        team_size (int): number of agents in each game
        n_games (int): number of games to sample
        verbose (bool, optional): this will print the agent, spy, and combination counters. Defaults to False.
        max_same_agent (int, optional): maximum number of the same agent in a game. Defaults to 2.

    Returns:
        list[tuple[list[str], int]]: list of tuples containing the agents in each game and the spy index
    """

    # Initialize counters for fairness
    agent_count = Counter({agent: 0 for agent in agent_names})
    spy_count = Counter({agent: 0 for agent in agent_names})
    game_agents_count = Counter()

    output = []
    for _ in range(n_games):
        agent_count_bias = min(agent_count.values())
        spy_count_bias = min(spy_count.values())
        game_agents = []
        while len(game_agents) < team_size:
            # Select a random agent with weighting based on fairness
            agent = random.choices(
                agent_names,
                weights=[
                    1 / (agent_count[agent] - agent_count_bias + 1)
                    for agent in agent_names
                ],
            )[0]

            # Ensure at most max_same_agent of the same agent in a game
            if game_agents.count(agent) < max_same_agent:
                game_agents.append(agent)
                agent_count[agent] += 1

        # Randomly assign one agent as the spy, ensuring fairness
        spy_ids = random.sample(range(team_size), n_spies)
        spy_names = [game_agents[i] for i in spy_ids]

        for spy_name in spy_names:
            spy_count[spy_name] += 1
        
        game_agents_count["".join(sorted(game_agents))] += 1

        output.append((game_agents, spy_ids))

    if verbose:
        print(agent_count)
        print(spy_count)
        print(game_agents_count)

    return output


def import_agents_from_files(glob_pattern: str) -> None:
    """loads an agent from a file and adds it to the agent registry
    Correctly handles imports, prioritizing the working directory over the submission directory
    Args:
        file_path (str): the path to the agent's submission file
    """
    for file in glob(glob_pattern, recursive=True):
        # submitted agents will prioritize files in the working directory, then the submission directory
        # name = file
        name = f"agent_{dirname(file).replace('/', '_')}_{file.split('/')[-1].replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(name, file)  # type: ignore
        module = importlib.util.module_from_spec(spec)  # type: ignore
        orig_path = sys.path
        sys.modules[name] = module
        sys.path.insert(1, dirname(file))
        spec.loader.exec_module(module)
        sys.path = orig_path


def relative_path_decorator(cls):
    """A class decorator that changes the working directory to the directory where the class is defined
    This is needed for submissions that load from files"""

    class_dir = os.path.dirname(sys.modules[cls.__module__].__file__)

    # Wrap all methods
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):

            @functools.wraps(attr_value)
            def wrapped_method(*args, original_method=attr_value, **kwargs):
                original_dir = os.getcwd()
                try:
                    # change working dir
                    os.chdir(class_dir)
                    result = original_method(*args, **kwargs)
                    return result
                finally:
                    # restore working dir
                    os.chdir(original_dir)

            # Handle static and class methods
            if isinstance(attr_value, staticmethod):
                wrapped_method = staticmethod(wrapped_method)
            elif isinstance(attr_value, classmethod):
                wrapped_method = classmethod(wrapped_method)

            # Update method
            setattr(cls, attr_name, wrapped_method)

    return cls


def rate_limit(requests_per_second: int):
    """A decorator to rate limit a function to a certain number of requests per second
    Args:
        requests_per_second (int): the number of requests per second
    """

    def decorator(func):
        last_time = 0
        request_lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with request_lock:
                nonlocal last_time
                await asyncio.sleep(
                    last_time + (1 / requests_per_second) - time.monotonic()
                )
                last_time = time.monotonic()
            while True:
                wait_time = 0.1
                try:
                    return await func(*args, **kwargs)
                except RateLimitError:
                    # exponential backoff
                    await asyncio.sleep(wait_time)
                    wait_time *= 1.2

        return wrapper

    return decorator


VOICES = [
    ("en", "com.au"),
    ("en", "co.uk"),
    ("en", "us"),
    ("en", "co.in"),
    # ("en", "com.ng"),
    # ("fr", "fr"),
    # ("fr", "ca"),
    # ("pt", "com.br"),
    # ("pt", "pt"),
    # ("es", "com.mx"),
    # ("es", "es"),
]
PITCH_SHIFTS = [0.7, 0.85, 1.0, 1.15]


def text_to_speech(
    text, voice: tuple[str, str] = ("en", "com.au"), pitch_shift_factor: float = 1.0
) -> tuple[np.ndarray, int]:
    """Convert text to speech using Google Text-to-Speech

    Args:
        text (_type_): The text to convert to speech
        voice (tuple[str, str]): The voice to use for the speech, as a tuple of language and region
        pitch_shift_factor (float): The factor by which to shift the pitch

    Returns:
        tuple[np.ndarray, int]: The numpy array of the audio and the sample rate
    """
    lang, tld = voice

    # call api
    tts = gTTS(text=text, lang=lang, tld=tld)
    while True:
        try:
            tts.save(".temp.mp3")
            break
        except gtts.tts.gTTSError:
            continue

    # load into parselmouth object
    sound = parselmouth.Sound(".temp.mp3")
    os.remove(".temp.mp3")
    sr = int(sound.sampling_frequency)

    # Adjust pitch
    if pitch_shift_factor != 1.0:
        manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(
            pitch_tier,
            "Multiply frequencies",
            sound.xmin,
            sound.xmax,
            pitch_shift_factor,
        )
        call([pitch_tier, manipulation], "Replace pitch tier")
        sound = call(manipulation, "Get resynthesis (overlap-add)")

    x = np.squeeze(sound.values.T) * (2**15 - 1)
    x = x.astype(np.int16)

    return x, sr


def get_voice_and_ps(player_name: str) -> tuple[tuple[str, str], float]:
    """Get the voice and pitch shift for a player based on their name

    Args:
        player_name (str): The player's name

    Returns:
        tuple[tuple[str, str], float]: The voice and pitch shift
    """
    voice = VOICES[hash(player_name) % len(VOICES)]
    pitch_shift = PITCH_SHIFTS[hash(player_name) % len(PITCH_SHIFTS)]
    return voice, pitch_shift


if __name__ == "__main__":
    sample_agents(["a", "b", "c", "d", "e", "f"], 4, 100, True)

    import pygame

    os.environ["SDL_AUDIODRIVER"] = "coreaudio"

    for ps in PITCH_SHIFTS:
        for voice in VOICES:
            print(f"Voice {voice}")
            text = "Hello, this is an AI voice generated from text."
            audio, sr = text_to_speech(text, voice, ps)

            pygame.mixer.pre_init(frequency=sr, channels=1, allowedchanges=0)
            pygame.init()
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
