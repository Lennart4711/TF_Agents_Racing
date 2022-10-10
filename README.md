# Agents Racing
## Environment
Created a gym environment to train the model on. It contains a car with n sensors and borders that mark the track. The goal is to be able to drive inside this track. A track editor is provided [here](level_editor.py). You can try it yourself by running [play.py](play.py)

## NEAT
Implemented a form of the genetic algorithm NEAT described in [this paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) in [neat.py](neat.py)

## TODO
- Change the [reinforce_agent](reinforce_agent.py)
- Better reward function than driven distance because it leads to turning in one spot
- Implement more algorithms
- Multiple agents in one environment
- Better folder structure and order by changing how imports are handled
