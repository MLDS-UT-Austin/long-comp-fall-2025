from simulation import *
from util import *

# This file will let you run games ####################################


if __name__ == "__main__":
    # Select the NLP model to use ####################################################

    # 1: Use Together.ai for Llama and BERT
    nlp = NLP(llm=Llama(), embedding=BERTTogether())

    # 2: Use Together.ai for Llama but run BERT locally
    # nlp = NLP(llm = Llama(), embedding=BERTLocal(batch_size=8))

    # 3: Output the same string for every question and return a 0 array for every embedding
    # nlp = NLP(llm=DummyLLM(), embedding=DummyEmbedding())

    # Load agents from specific files ####################################################
    # for you to run your agent
    import_agents_from_files("submission.py")
    # for you to run an example agent
    import_agents_from_files("example agents/agents.py")
    # for us to run your agents
    import_agents_from_files("submissions/*/submission.py")

    # Run a single game ####################################################
    # Feel free to edit the players, duplicate players are allowed
    game = Game(
        player_names=["Team Name Here", "MLDS 0", "MLDS 1", "MLDS 2", "MLDS 3", "MLDS 4"],
        nlp=nlp,
        n_rounds=20,
    )
    game.play()
    print(game)
    print("Scores:", game.get_scores())
    print("Percent Right Votes:", game.get_percent_right_votes())
    print("Game Duration:", len(game.rounds))

    conv = game.save_conversation("conversation.csv")

    # game.pregenerate_audio()
    # game.render()
    # game.save_audio("game.wav")

    # Run multiple games with randomly sampled agents ####################################################
    sim = Simulation(
        nlp, agent_names=["Team Name Here", "MLDS 0", "MLDS 1", "MLDS 2", "MLDS 3", "MLDS 4"], team_size=8
    )
    sim.validate_agents()
    sim.run(n_games=10)
    # average scores of all agents
    print("Average Scores:", sim.get_scores().mean(axis=0))
    print("Average Percent Right Votes:", sim.get_percent_right_votes().mean(axis=0))
    print("Game End Type Frequency:", sim.get_game_endtype_freq())
    print("Game Duration Frequency:", sim.get_game_duration_freq())

    # sim.visualize_scores()
    # sim.save_visualization("simulation.mp4")
    # sim.pickle_games()
