# Section Description
This section contains all code used to develop the final version of the DDPG algorithm. Additionally, models and rewards have been saved (in their respective sub-folders): best models are saved in `best_models` (both actor and critic models, together with the target networks, as DDPG is an actor-critic approach).

# Code Usage
- To run the best saved model, simply open `evaluation.py` and run this.
- To train a new model from scratch, open `training.py` and run this. 1500 training episodes will be run, and every 100 episodes the reward will be saved to a CSV file in `rewards/`, and the corresponding model will also be saved. If the reward at the end of an episode is greater than the current best reward (initialised to 0 prior to training), the model is saved to `best_models`.
