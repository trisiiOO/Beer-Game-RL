# Import necessary libraries
import pandas as pd
import numpy as np
import logging
import time

# Record the start time
start_time = time.time()


# Set up logging  
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(lineno)d - %(message)s')
logger = logging.getLogger()


class SupplyChainEnv:
    def __init__(self, initial_inventory, holding_costs, penalty_costs, customer_demand, lead_times):
        """
        Initialize the environment for the beer game.

        Args:
            initial_inventory (list): Initial inventory levels for each actor in the supply chain.
            holding_costs (list): Holding costs per unit for each level in the supply chain.
            penalty_costs (list): Penalty costs per unit for each level in the supply chain.
            customer_demand (list): List of customer demand values over the time horizon.
            lead_times (list): List of lead times the same for each level.
            current_time (int): The current time step, starting at zero.
            reset (method): adding to init, to not have duplicate code.
        """
        self.initial_inventory = initial_inventory
        self.holding_costs = holding_costs
        self.penalty_costs = penalty_costs
        self.customer_demand = customer_demand
        self.lead_times = lead_times
        self.current_time = 0
        self.reset() 


    def reset(self):
        """
        Reset the environment to its initial state.
        This method sets up the initial conditions of the simulation, including inventory levels and backlogs.
        """
        self.inventory_levels = self.initial_inventory.copy()
        self.order_backlog = [0] * len(self.initial_inventory)
        self.current_time = 0

        # Initialize required inventory list for each agent
        # plus 1 to account for customer demand
        self.required_inventory = [0] * (len(self.initial_inventory)+1)
        
        # Initialize pending orders list for each agent
        self.pending_orders = [[] for _ in range(len(self.initial_inventory))]
        
        # Add initial pending orders of 4 units with lead times 0 and 1 for each agent
        for i in range(len(self.initial_inventory)):
            self.pending_orders[i].append((1, 4))  # Lead time 1
            self.pending_orders[i].append((2, 4))  # Lead time 2
        return self.get_state()


    def get_state(self):
        """
        Get the current state of the environment.
        The state is represented by coded inventory levels.

        Returns:
            tuple: A tuple representing the coded inventory levels.
        """
        # subtracting the backlog from the inventory levels and then coding the state
        return self.code_state(tuple(i - b for i, b in zip(self.inventory_levels, self.order_backlog)))


    def code_state(self, state):
        """
        Encode the state according to predefined ranges into discrete categories.

        Args:
            state (tuple): The actual state as a tuple of inventory levels.

        Returns:
            tuple: The coded state.
        """
        
        # iterating over each element of the state vector and encoding it
        coded_state = []
        for s in state:
            if s <= -6:
                coded_state.append(1)
            elif -6 < s <= -3:
                coded_state.append(2)
            elif -3 < s <= 0:
                coded_state.append(3)
            elif 0 < s <= 3:
                coded_state.append(4)
            elif 3 < s <= 6:
                coded_state.append(5)
            elif 6 < s <= 10:
                coded_state.append(6)
            elif 10 < s <= 15:
                coded_state.append(7)
            elif 15 < s <= 20:
                coded_state.append(8)
            else:
                coded_state.append(9)
        return tuple(coded_state)


    def get_reward(self):
        """
        Calculate the reward based on the current state.
        The reward is negative, representing costs.

        Returns:
            int: The negative cost, calculated as the sum of holding costs and penalty costs.
        """
        holding_cost = sum(
            [self.holding_costs[i] * max(0, self.inventory_levels[i]) for i in range(len(self.inventory_levels))])
        penalty_cost = sum(
            [self.penalty_costs[i] * max(0, self.order_backlog[i]) for i in range(len(self.inventory_levels))])
        return -(holding_cost + penalty_cost)


    def step(self, action):
        """
        Perform a single time step in the environment.

        Args:
            action (list): A list of actions for each agent in the supply chain.

        Returns:
            tuple: The new state and the reward obtained.
        """
        
        # Delivery Process
        for i in range(len(self.inventory_levels)):
            # Get orders that should be delivered at the current time
            orders_to_deliver = [order for order in self.pending_orders[i] if order[0] <= self.current_time]

            # Process each order
            for order in orders_to_deliver:
                delivery_time, amount = order
                self.inventory_levels[i] += amount
                # Remove the fulfilled order from the pending orders list
                self.pending_orders[i].remove(order)


        # Ordering Process
        for i in range(len(self.inventory_levels)):
            if i == 0:
                # For the retailer, set the required inventory directly to the demand
                self.required_inventory[i] = self.customer_demand[self.current_time]
            else:
                # For other agents, calculate the required inventory
                X = self.required_inventory[i-1]
                Y = action[i-1]
                self.required_inventory[i] = X + Y


            # Prioritize fulfilling backorders first
            backorder_fulfilled = 0
            if self.order_backlog[i] > 0:
                if self.inventory_levels[i] >= self.order_backlog[i]:
                    backorder_fulfilled = self.order_backlog[i]
                    self.inventory_levels[i] -= backorder_fulfilled
                    self.order_backlog[i] = 0
                else:
                    backorder_fulfilled = self.inventory_levels[i]
                    self.order_backlog[i] -= backorder_fulfilled
                    self.inventory_levels[i] = 0


            # Attempt to fulfill the current required inventory
            if self.inventory_levels[i] >= self.required_inventory[i]:
                order_fulfilled = self.required_inventory[i]
            else:
                order_fulfilled = self.inventory_levels[i]
                # Adding demand that couldn't be fullfilled to the backlog
                self.order_backlog[i] += self.required_inventory[i] - order_fulfilled

            # Total fulfilled order including backorders and current demand
            total_fulfilled = backorder_fulfilled + order_fulfilled
            self.inventory_levels[i] -= order_fulfilled

            # Update supplier's inventory and add to pending orders list
            # Skip adding to pending orders if i == 0
            if i != 0:
                self.pending_orders[i-1].append((self.current_time + self.lead_times[self.current_time], total_fulfilled))

            # Special handling for the factory (Agent 3)
            if i == 3:
                # The factory can always produce the required amount
                self.required_inventory[i+1] = self.required_inventory[i] + action[i]
                # Record production in pending orders with the appropriate lead time
                self.pending_orders[i].append((self.current_time + self.lead_times[self.current_time], self.required_inventory[i+1]))


        # Delivery Process again, to account for lead times of zero
        for i in range(len(self.inventory_levels)):
            # Get orders that should be delivered at the current time
            orders_to_deliver = [order for order in self.pending_orders[i] if order[0] <= self.current_time]

            # Process each order
            for order in orders_to_deliver:
                delivery_time, amount = order
                self.inventory_levels[i] += amount
                # Remove the fulfilled order from the pending orders list
                self.pending_orders[i].remove(order)

        self.current_time += 1

        return self.get_state(), self.get_reward()


#-----------------------------------------------------------------------------------------------------------------------
    

class QLearning:
    def __init__(self, actions, state_space, alpha=0.17, gamma=1, epsilon_start=0.98, epsilon_end=0.1,
                 epsilon_final=0.02, max_iterations=1, num_agents=4):
        """
        Initialize the Q-learning agent.

        Args:
            Q_tables (List of Dictionaries): Q-tables for each agent. -> makeing decisions independently
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_final (float): Final exploration rate within each period.
            max_iterations (int): Maximum number of iterations for training.
            actions (list): List of possible actions/action space.
            state_space (list): List of all possible states.
            num_agents (int): Number of agents in the supply chain, default 4, see beer game.
        """
        # Initialize Q-table as a List of dictionaries
        self.Q_tables = [{} for _ in range(num_agents)]  # One Q-table per agent
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon_start = epsilon_start # Initial exploration rate
        self.epsilon_end = epsilon_end  # Final exploration rate
        self.epsilon_final = epsilon_final # Final exploration rate within each period
        self.max_iterations = max_iterations # Maximum number of iterations for training
        self.actions = actions # List of possible actions
        self.state_space = state_space # List of all possible states
        self.num_agents = num_agents # Number of agents in the supply chain


    def choose_action(self, state, epsilon):
        """
        Choose an action based on the current state using an epsilon-greedy policy.

        Args:
            state (tuple): The current state.
            epsilon (float): The current exploration rate.

        Returns:
            tuple: The chosen actions as a vector.
        """

        action_vector = []
        # making sure each agent chooses his action on his own, without taking the other agents' actions into account
        for agent in range(self.num_agents):

            # Exploration: choose a random action
            if np.random.rand() < epsilon:
                action = int(np.random.choice(self.actions))
            
            # Exploitation: choose the best action based on the agent's Q-table
            else:
                if state not in self.Q_tables[agent] or not self.Q_tables[agent][state]:
                    action = int(np.random.choice(self.actions))
                else:
                    action = max(self.Q_tables[agent][state], key=self.Q_tables[agent][state].get)
            action_vector.append(action)

        return tuple(action_vector)


    def choose_greedy(self, state):
        """
        Choose the best action based on the current state using a greedy policy.

        Args:
            state (tuple): The current state.

        Returns:
            tuple: The chosen actions as a vector.
        """
        # Exploit: choose the action with the highest Q-value
        action_vector = []

        for agent in range(self.num_agents):
            if state not in self.Q_tables[agent] or not self.Q_tables[agent][state]:
                action = int(np.random.choice(self.actions))
            else:
                action = max(self.Q_tables[agent][state], key=self.Q_tables[agent][state].get)
            action_vector.append(action)
        return tuple(action_vector)


    def update_Q(self, state, action_vector, reward, next_state):
        """
        Update the Q-value for the given state-action pair.

        Args:
            state (tuple): The current state.
            action (tuple): The actions taken.
            reward (int): The reward received.
            next_state (tuple): The next state.
        """

        # Update the Q-value using the Q-learning formula:
        # Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        # This incorporates the immediate reward and the estimated optimal future reward,
        # Adjusted by the learning rate (alpha) and the discount factor (gamma)
        
        # Update Q-table for each agent
        for agent in range(self.num_agents):
            action = action_vector[agent]
            
            # Find the maximum Q-value for the next_state over all possible actions for the current agent, defaulting to 0.0 if empty
            max_q_next = max(self.Q_tables[agent][next_state].values(), default=0.0)            

            # Retrieve the current Q-value for the state-action pair, defaulting to 0 if not present
            old_value = self.Q_tables[agent][state].get(action, 0)
            
            # Calculate the TD target
            td_target = reward + self.gamma * max_q_next
            
            # Calculate the TD error
            td_error = td_target - old_value
            
            # Update the Q-value using the Q-learning update rule
            new_value = old_value + self.alpha * td_error
            self.Q_tables[agent][state][action] = new_value


    def train(self, env):
        """
        Train the Q-learning agent.

        Args:
            env (SupplyChainEnv): The environment.

        Returns:
            list: Logs of the training process.
        """

        logs = []
        # Linearly decrease epsilon within the iteration - increasing exploitation
        epsilon_decrement_outer = (self.epsilon_start - self.epsilon_end) / self.max_iterations
        new_epsilon_start = self.epsilon_start
        for iteration in range(self.max_iterations):
            state = env.reset()
            t = 0
            episode_log = []
            # Linearly decrease epsilon within the period - increasing exploitation
            epsilon_decrement = (new_epsilon_start - self.epsilon_final) / time_horizon
            epsilon = new_epsilon_start
            reward = 0
            while t < time_horizon:
                # a) Select an action using the epsilon-greedy policy
                action_vector = self.choose_action(state, epsilon)

                # Log for each timestamp during training
                episode_log.append((state, action_vector, reward, env.customer_demand[t], list(env.inventory_levels), list(env.order_backlog)))
                
                # b) Caclulate the next state and the reward - includes doing action & doing state vector
                next_state, reward = env.step(action_vector)

                # Update Q-table for each agent
                for agent in range(self.num_agents):
                    action = action_vector[agent]

                    # Fill Q-table with default values if not present
                    if state not in self.Q_tables[agent]:
                        self.Q_tables[agent][state] = {}
                    if action not in self.Q_tables[agent][state]:
                        self.Q_tables[agent][state][action] = 0

                    if next_state not in self.Q_tables[agent]:
                        self.Q_tables[agent][next_state] = {}


                # c) Update Q-table, using next state and actual reward
                self.update_Q(state, tuple(action_vector), reward, next_state)

                # Further decrease epsilon within the period - increasing exploitation
                epsilon -= epsilon_decrement

                # d) update state
                state = next_state

                # e) update time
                t += 1
            
            # Loging the final state of each episode
            episode_log.append((state, action_vector, reward, 0, list(env.inventory_levels), list(env.order_backlog)))
            logs.append(episode_log)

            # Linearly decrease epsilon within the iteration - increasing exploitation
            new_epsilon_start -= epsilon_decrement_outer

        # Run simulation after training is complete
        simulation_log = self.simulation(env)

        return logs, simulation_log
    

    def simulation(self, env):
        """
        Run the greedy policy on the learned Q-table.
        No more exploration, only taking the best action according to the Q-table.

        Args:
            env (SupplyChainEnv): The environment.

        Returns:
            list: Logs of the simulation process.
        """

        state = env.reset()
        t = 0
        simulation_log = []
        reward = 0
        while t < time_horizon:
            # Select the best action according to the greedy policy
            action = self.choose_greedy(state)
            simulation_log.append((state, action, reward, env.customer_demand[t], list(env.inventory_levels), list(env.order_backlog)))

            # Calculate the next state and the reward
            next_state, reward = env.step(action)
            
            # Log after each step during simulation

            state = next_state
            t += 1
        
        # Log the final state period 35
        simulation_log.append((state, action, reward, 0, list(env.inventory_levels), list(env.order_backlog)))

        return simulation_log




# -----------------------------------------------------------------------------------------------------------------------
# Test Data & Parameters
# Test problems data, as can found in the paper
test_problems = {
    "main": {
        "customer_demand": [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15, 12, 15, 4, 12, 3, 13, 10, 15, 15, 3,
                            11, 1, 13, 10, 10, 0, 0, 8, 0, 14],
        "lead_times": [2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4, 0,
                       3, 3, 4]
    },
    "test1": {
        "customer_demand": [5, 14, 14, 13, 2, 9, 5, 9, 14, 14, 12, 7, 5, 1, 13, 3, 12, 4, 0, 15, 11, 10, 6, 0, 6, 6, 5,
                            11, 8, 4, 4, 12, 13, 8, 12],
        "lead_times": [2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4, 0,
                       3, 3, 4]
    },
    "test2": {
        "customer_demand": [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15, 12, 15, 4, 12, 3, 13, 10, 15, 15, 3,
                            11, 1, 13, 10, 10, 0, 0, 8, 0, 14],
        "lead_times": [4, 2, 2, 0, 2, 2, 1, 1, 3, 0, 0, 3, 3, 3, 4, 1, 1, 1, 3, 0, 4, 2, 3, 4, 1, 3, 3, 3, 0, 3, 4, 3,
                       3, 0, 3]
    },
    "test3": {
        "customer_demand": [13, 13, 12, 10, 14, 13, 13, 10, 2, 12, 11, 9, 11, 3, 7, 6, 12, 12, 3, 10, 3, 9, 4, 15, 12,
                            7, 15, 5, 1, 15, 11, 9, 14, 0, 4],
        "lead_times": [4, 2, 2, 0, 2, 2, 1, 1, 3, 0, 0, 3, 3, 3, 4, 1, 1, 1, 3, 0, 4, 2, 3, 4, 1, 3, 3, 3, 0, 3, 4, 3,
                       3, 0, 3]
    }
}


# Defined test parameters
initial_inventory = [12, 12, 12, 12]
holding_costs = [1, 1, 1, 1]
penalty_costs = [2, 2, 2, 2]
time_horizon = 35 # Length of customer demand list
actions = [0, 1, 2, 3]  # Range for Y in X+Y rule according to the paper / action space
state_space = [(i, j, k, l) for i in range(1, 10) for j in range(1, 10) for k in range(1, 10) for l in
               range(1, 10)] # Each agent has 9 possible states


# Initialize environment and Q-learning agent for the main test problem
env = SupplyChainEnv(initial_inventory, holding_costs, penalty_costs, test_problems["main"]["customer_demand"],
                     test_problems["main"]["lead_times"])


# define max iterations
agent = QLearning(actions, state_space, max_iterations=100000) # Best results were found using 100k iterations


# Train the agent + + logggs
logs, simulation_log  = agent.train(env)




# -----------------------------------------------------------------------------------------------------------------------
# Analysis of the results

# Defining the print_logs function 
def print_logs(logs, num_episodes=1):
    for episode_index in range(-num_episodes, 0):
        episode_log = logs[episode_index]
        df = pd.DataFrame(episode_log, columns=['State', 'Action', 'Reward', 'Demand', 'Inventory Levels', 'Order Backlog'])
        print(f"Episode {len(logs) + episode_index + 1}")
        print(df)
        print("\n")


# List to store total rewards for each episode
episode_rewards = []

# Iterate over each episode and sum the rewards, calculating the total rewards per episode
for episode in logs:
    total_episode_reward = sum(log[2] for log in episode)
    episode_rewards.append(total_episode_reward)

#logging.debug(f"Total rewards per episode: {episode_rewards}")

# Print the total reward of the final episode
if episode_rewards:
    print(f"Total reward for the final episode: {episode_rewards[-1]}")


# Print the logs of the last episode
print_logs(logs, num_episodes=1)


# List to store total rewards for the greedy policy run.
episode_rewards_sim = []


# Calculate the total cost of the simulation & print simulation logs
total_episode_reward_sim = sum(log[2] for log in simulation_log)
episode_rewards_sim.append(total_episode_reward_sim)

print(f"Total reward for Simulation runs: {episode_rewards_sim}")

df = pd.DataFrame(simulation_log, columns=['State', 'Action', 'Reward', 'Demand', 'Inventory Levels', 'Order Backlog'])
print(df)
print("\n")


# Record the end time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time

# Print the execution time
print(f"Total execution time: {execution_time:.2f} seconds")


# -----------------------------------------------------------------------------------------------------------------------

# To save the logs to a CSV file
# simulation_df = pd.DataFrame(simulation_log, columns=['State', 'Action', 'Reward', 'Demand', 'Inventory Levels', 'Order Backlog'])
# simulation_df.to_csv('simulation_logs_100k_standXY.csv', index=False)

