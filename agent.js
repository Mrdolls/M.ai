class QLearningAgent {
    constructor(actions, alpha = 0.1, gamma = 0.9, epsilonStart = 1.0, epsilonDecay = 0.995, epsilonMin = 0.01) {
        this.qTable = new Map(); // Using a Map for sparse Q-table (state can be any string)
        this.actions = actions; // Array of possible actions [0, 1, 2, 3, 4]
        this.alpha = alpha; // Learning rate
        this.gamma = gamma; // Discount factor
        this.epsilon = epsilonStart; // Exploration rate
        this.epsilonStart = epsilonStart;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
    }

    // Get Q-value for a state-action pair. If not exists, initialize to 0.
    getQValue(state, action) {
        if (!this.qTable.has(state)) {
            this.qTable.set(state, new Array(this.actions.length).fill(0.0));
        }
        return this.qTable.get(state)[action];
    }

    // Choose an action based on epsilon-greedy policy
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
            // Explore: choose a random action
            return this.actions[Math.floor(Math.random() * this.actions.length)];
        } else {
            // Exploit: choose the best action based on Q-values
            if (!this.qTable.has(state)) {
                // If state is unknown, initialize Q-values (or could choose randomly)
                this.qTable.set(state, new Array(this.actions.length).fill(0.0));
                // And then pick a random action as we have no info
                return this.actions[Math.floor(Math.random() * this.actions.length)];
            }
            const qValues = this.qTable.get(state);
            let maxQ = -Infinity;
            let bestActions = [];

            for (let i = 0; i < qValues.length; i++) {
                if (qValues[i] > maxQ) {
                    maxQ = qValues[i];
                    bestActions = [this.actions[i]];
                } else if (qValues[i] === maxQ) {
                    bestActions.push(this.actions[i]);
                }
            }
            // If multiple actions have the same max Q-value, pick one randomly
            return bestActions[Math.floor(Math.random() * bestActions.length)];
        }
    }

    // Update Q-value for a state-action pair
    updateQValue(state, action, reward, nextState) {
        const oldQValue = this.getQValue(state, action); // Ensures state is initialized if new

        // Get max Q-value for the next state (max_a' Q(s', a'))
        let maxNextQ = 0;
        if (this.qTable.has(nextState)) {
            const nextQValues = this.qTable.get(nextState);
            maxNextQ = Math.max(...nextQValues);
        }
        // If nextState is not in qTable, its Q-values are effectively 0, so maxNextQ is 0.

        // Q-learning formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        const newQValue = oldQValue + this.alpha * (reward + this.gamma * maxNextQ - oldQValue);

        // Ensure state entry exists before trying to set action's Q-value
        if (!this.qTable.has(state)) {
            this.qTable.set(state, new Array(this.actions.length).fill(0.0));
        }
        this.qTable.get(state)[action] = newQValue;
    }

    // Decay epsilon
    decayEpsilon() {
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
    }

    resetEpsilon() {
        this.epsilon = this.epsilonStart;
    }

    setParameters(alpha, gamma, epsilonStart, epsilonDecay) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilon = epsilonStart; // Reset current epsilon to new start
        this.epsilonDecay = epsilonDecay;
        // EpsilonMin is not currently settable by user, could be added.
    }

    clearQTable() {
        this.qTable.clear();
        console.log("Q-Table cleared.");
    }

    getQTable() {
        // Convert Map to a serializable object for saving if needed
        const serializableQTable = {};
        for (const [state, values] of this.qTable) {
            serializableQTable[state] = values;
        }
        return serializableQTable;
    }

    loadQTable(loadedData) {
        this.qTable.clear();
        for (const state in loadedData) {
            this.qTable.set(state, loadedData[state]);
        }
        console.log("Q-Table loaded.");
    }
}
