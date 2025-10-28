<img src="media/readme_image.jpg" alt="spyfall" width="250" align="right" style="padding: 50px;"/>

# Long Competition 2025: Spyfall

This is the repository for the 2025 UT Austin MLDS Long Competition where teams will be competing in a game of Spyfall. Teams will submit agents that can use LLMs and embedding models to ask/answer questions and determine the spy/location. The goal of this Long Competition is to show your LLM prompting/natural language processing skills.

## Game Rules

**<u>Setup</u>**: Each game is played with 4-12 players. All players are given the location except for the spy.

**<u>Spy Objective</u>**: The spy must figure out the location without revealing their identity.

**<u>Non-Spy Objective</u>**: Players must figure out who the spy is.

**<u>Gameplay</u>**: Each game consists of a fixed number of rounds. In each round, the following happens:

1. **<u>Questioning</u>**: A random player starts by asking another player a question about the location. The player who answers the question will be the one to ask the question in the next round. You can ask a question to any player except the player who asked you a question the previous round.

   Ex: A --> B --> A is not allowed.
2. **<u>Questioning Analysis</u>**: All players are given time to analyze the question/answer.
3. **<u>Guessing</u>**: The spy may guess the location. This will end the game.
4. **<u>Accusation</u>**: Players may accuse another player of being the spy. Successfully indicting a player will end the game. For a player to be indicted, the following conditions must be met:
   * A majority of the players must accuse *any* player.
   * One player must get a plurality of the votes. If a tie occurs, nothing happens.

    Ex: If 2 players accuse player A, 1 player accuses player B, 1 player accuses player C, and 3 players do not vote, player A is successfully indicted.

5. **<u>Voting Analysis</u>**: Players can see who voted for who and are given time to analyze the votes.

**<u>The game ends when</u>**:

* The spy guesses the location.
* A player is indicted of being the spy.
* All rounds are completed.

**<u>Scoring</u>**:

# TODO fixme: scoring details
* **<u>Spy Victory</u>**: The spy earns 2 points if no one is indicted of being the spy, 4 points if a non-spy player is indicted of being the spy, and 4 points if the spy stops the game and successfully guesses the location.
* **<u>Non-Spy Victory</u>**: Each non-spy player earns 1 point.

## Getting Started

**<u>Setting up the Code</u>**:

To run this project locally with conda, run the following commands. This may take a while.

``` bash
conda create -n long_comp python==3.10.13 -y
conda activate long_comp
pip install -r requirements.txt
```

If you do not have conda installed, you can install it [here](https://docs.anaconda.com/miniconda/). Additionally, if you use VSCode, the VSCode Extension `Python Environment Manager` by Don Jayamanne is nice for managing and selecting default conda environments.

If you use additional packages, please add them to `requirements.txt`.

*Note: this repository is thoroughly type-annotated. To catch type errors, you can install the VSCode Extension `Mypy Type Checker` by Microsoft.*

**<u>Setting up Your API Key</u>**:

We will be using [Google Vertex AI](https://console.cloud.google.com/freetrial/?redirectPath=/vertex-ai) which offers $300 in free credits. A credit card is required but only to verify identity. 

***You will not be charged unless you explicitly upgrade your account.***

1. Login to your Google account
2. Enter your credit card information (again, you will not be charged)
3. Click agree/enable to all pop-ups
4. Click on "My First Project" at the top left and select "New project"
5. Create a new project called "long-comp-fall-2025"
6. Install gcloud cli and login with the following commands. Click "enter" when prompted for a path, "y" to update your $PATH, and "n" to install Python 3.12.

``` bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
# gcloud init
gcloud auth application-default login
gcloud auth application-default set-quota-project long-comp-fall-2025
```

**<u>Running Games/Simulations</u>**:

See `main.py` for an example of how to run games and simulations.

*Note: by default, a dummy llm and embedding model are selected. You can change this at the top of `main.py`.*

**<u>Submitting Your Agent</u>**:

Write your agent in `submission.py` and use `@register_agent(<team name here>)` to register your agent under your team name. Also, please put your team member's name/emails/EIDs in a comment at the top. You can name your class anything you want. Commit and push `submission.py` and any other files you added to the GitHub Classroom repository.

## Rules

* Besides `submission.py`, `requirements.txt`, and any added files, do not modify any other files in the repository. For instance, do not change `data.py`. If you need to store data, use a separate file.
* Prompt injections are allowed.
* You may not use any NLP models outside what is provided.
