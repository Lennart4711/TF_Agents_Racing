from neural_network import NeuralNetwork
from environment import Environment
from dataclasses import dataclass
import pickle


@dataclass
class Agent:
    nn: NeuralNetwork
    reward: float = 0
    env: Environment = Environment(has_surface=True)


def visualize(nn: NeuralNetwork):
    env = Environment()
    step = env.reset()
    reward = 0
    while not step.is_last():
        action = nn.forward(step.observation)
        action[0] -= 0.5
        step = env.step(action)
        env.render(telemetry=True)
        reward += step.reward
    print(f"VIs finished with reward {reward}")


def run_agent(agent: Agent):
    step = agent.env.reset()
    while not step.is_last() and agent.env.state < 10000:
        action = agent.nn.forward(step.observation)
        action[0] -= 0.5
        step = agent.env.step(action)
        agent.reward += step.reward
        # agent.env.render(telemetry=True)
        if agent.env.state % 100 == 0:
            print(f"Agent at state {agent.env.state} with reward {agent.reward}")

    
    return agent


agent_count = 7
mutation_rate = 0.2


load = False
if load:
    # Load agents from file
    with open("best_agent.pkl", "rb") as f:
        nn = pickle.load(f)

    agents = [Agent(nn=nn) for _ in range(agent_count)]
else:
    agents = [Agent(NeuralNetwork([8, 16, 16, 6, 2])) for _ in range(agent_count)]

try:
    while True:

        agents = [run_agent(agent) for agent in agents]  # Parallelize this
        agents.sort(key=lambda agent: agent.reward, reverse=True)
        print(agents[0].reward)
        print(agents[-1].reward)


        visualize(agents[0].nn)
        new_agents = []
        # Append slightly mutated versions of the best 10% of agents
        for agent in agents[: int(agent_count * 0.1)]:
            new_agents.append(Agent(agent.nn.reproduce(mutation_rate)))
        # Fill 10% with random agents
        for _ in range(int(agent_count * 0.1)):
            new_agents.append(Agent(NeuralNetwork([8, 5, 5, 2])))
        # Fill the rest with mutated versions of the best agent
        for _ in range(int(agent_count * 0.8)):
            new_agents.append(Agent(agents[0].nn.reproduce(mutation_rate)))
        new_agents[-1] = Agent(agents[0].nn.get_deep_copy())  # Keep the best agent
        agents = new_agents
except KeyboardInterrupt:
    # save the best agent
    with open("best_agent.pkl", "wb") as f:
        pickle.dump(agents[0].nn, f)
