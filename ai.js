// --- AI Configuration ---
// These are default values; they can be overridden by user inputs from the UI.
let LEARNING_RATE = 0.1;     // Alpha: How much new information overrides old information.
let DISCOUNT_FACTOR = 0.9;   // Gamma: Importance of future rewards.
let EXPLORATION_RATE = 1.0;  // Epsilon: Probability of choosing a random action (exploration).
let EPSILON_DECAY = 0.995;   // Rate at which epsilon decreases after each episode.
let MIN_EPSILON = 0.01;      // Minimum value for epsilon to ensure some exploration continues.

// --- Q-Table ---
// The Q-table stores the Q-values for each state-action pair.
// A state is represented by a string "x,y" (agent's coordinates).
// Actions are discrete: 0: up, 1: down, 2: left, 3: right, 4: wait.
let qTable = {}; // e.g., { "0,0": [q_up, q_down, q_left, q_right, q_wait], "0,1": [...] }
const NUM_ACTIONS = 5; // The total number of possible actions the agent can take.

// --- AI Helper Functions ---

// Retrieves Q-values for a given state from the Q-table.
// If the state is new, it initializes Q-values for it (typically to zeros).
function getQValues(state) {
    const stateKey = `${state.x},${state.y}`; // Create a string key for the state.
    if (!qTable[stateKey]) {
        // Initialize Q-values for all actions in this new state to 0.
        qTable[stateKey] = Array(NUM_ACTIONS).fill(0);
    }
    return qTable[stateKey];
}

// Selects an action for the agent based on the current state and epsilon-greedy strategy.
function chooseAction(state) {
    const qValues = getQValues(state); // Get Q-values for the current state.

    if (Math.random() < EXPLORATION_RATE) {
        // Explore: Choose a random action.
        return Math.floor(Math.random() * NUM_ACTIONS);
    } else {
        // Exploit: Choose the action with the highest Q-value.
        // If there's a tie, randomly choose among the best actions.
        let maxQ = -Infinity;
        let bestActions = [];
        for (let i = 0; i < qValues.length; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestActions = [i];
            } else if (qValues[i] === maxQ) {
                bestActions.push(i);
            }
        }
        return bestActions[Math.floor(Math.random() * bestActions.length)];
    }
}

// Updates the Q-value for a given state-action pair using the Q-learning formula.
// Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a'(Q(s',a')) - Q(s,a))
function updateQValue(state, action, reward, nextState) {
    const stateKey = `${state.x},${state.y}`;
    const currentQValues = getQValues(state); // Ensures state exists in qTable
    const oldQValue = currentQValues[action];

    const nextQValues = getQValues(nextState); // Q-values for the next state
    const maxNextQ = Math.max(...nextQValues); // Maximum Q-value for the next state

    // Apply the Q-learning update rule
    currentQValues[action] = oldQValue + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxNextQ - oldQValue);
    // No need to re-assign to qTable[stateKey] if currentQValues is a direct reference,
    // but explicit assignment is safer if getQValues ever returns a copy.
    qTable[stateKey] = currentQValues;
}

// --- Training Control Variables ---
let trainingIntervalId = null; // ID for setInterval, used to stop training.
let currentWave = 0;           // Current training episode/wave number.
let totalWaves = 1000;         // Total number of waves to train for (default, set by UI).
let bestBrain = null;          // Stores a deep copy of the Q-table with the highest score.
let bestScoreFromTraining = -Infinity; // Tracks the highest score achieved during training.
let stepsPerEpisode = 200;     // Maximum number of steps allowed per episode.

// Runs a single training episode.
// The agent interacts with the environment, learns, and updates its Q-table.
function runEpisode() {
    resetGame(); // Reset game environment (from game.js).
    let currentState = agent.getPosition(); // Get initial state from the game's agent.
    let episodeScore = 0; // Score accumulated in this current episode.

    // Loop for a fixed number of steps or until a terminal condition.
    for (let step = 0; step < stepsPerEpisode; step++) {
        const action = chooseAction(currentState); // Agent decides on an action.
        // gameStep executes the action, returns results (from game.js).
        const { reward, gameOver, newState, score: gameScore } = gameStep(action);

        updateQValue(currentState, action, reward, newState); // AI learns from the action.

        currentState = newState; // Move to the next state.
        episodeScore = gameScore;  // Update episode score from authoritative game score.

        // 'gameOver' from gameStep is usually false in this setup,
        // episode termination is primarily by stepsPerEpisode.
        if (gameOver) {
            break;
        }
    }

    // After the episode, update UI elements.
    document.getElementById('currentWave').textContent = currentWave;
    if (episodeScore > bestScoreFromTraining) {
        bestScoreFromTraining = episodeScore;
        document.getElementById('bestScore').textContent = bestScoreFromTraining;
        // Save a deep copy of the current Q-table as the "best brain".
        bestBrain = JSON.parse(JSON.stringify(qTable));
    }
}

// Starts the AI training process.
function startTraining() {
    if (trainingIntervalId) {
        console.log("Training is already in progress.");
        return;
    }

    // Fetch training parameters from the UI input fields.
    LEARNING_RATE = parseFloat(document.getElementById('learningRate').value);
    DISCOUNT_FACTOR = parseFloat(document.getElementById('discountFactor').value);
    EXPLORATION_RATE = parseFloat(document.getElementById('explorationRate').value); // Initial Epsilon
    EPSILON_DECAY = parseFloat(document.getElementById('epsilonDecay').value);
    totalWaves = parseInt(document.getElementById('trainingWaves').value);
    // Note: Hidden layers input is for potential DQN extension, not used in this Q-learning version.

    console.log(`Starting training with LR: ${LEARNING_RATE}, DF: ${DISCOUNT_FACTOR}, Eps: ${EXPLORATION_RATE}, Decay: ${EPSILON_DECAY}, Waves: ${totalWaves}`);

    // Reset AI state for a new training session.
    qTable = {};
    bestBrain = null;
    bestScoreFromTraining = -Infinity;
    currentWave = 0;
    document.getElementById('bestScore').textContent = bestScoreFromTraining; // Reset UI
    document.getElementById('currentEpsilon').textContent = EXPLORATION_RATE.toFixed(3);


    // Run episodes at intervals to allow UI to update and remain responsive.
    trainingIntervalId = setInterval(() => {
        if (currentWave < totalWaves) {
            currentWave++;
            runEpisode();
            // Decay exploration rate (epsilon) after each episode.
            EXPLORATION_RATE = Math.max(MIN_EPSILON, EXPLORATION_RATE * EPSILON_DECAY);
            document.getElementById('currentEpsilon').textContent = EXPLORATION_RATE.toFixed(3);
        } else {
            stopTraining(); // Automatically stop when all waves are completed.
            console.log("Training finished automatically.");
            alert("Training complete!");
        }
    }, 50); // Interval in milliseconds. Adjust for training speed vs. UI responsiveness.
}

// Stops the AI training process.
function stopTraining() {
    if (trainingIntervalId) {
        clearInterval(trainingIntervalId);
        trainingIntervalId = null;
        console.log("Training stopped by user or completion.");
        // Ensure UI reflects that training is stopped (e.g., button text).
        // This is handled in ui.js for user-initiated stops.
    }
}

// Returns the "best brain" (Q-table) found during training.
function getBestBrain() {
    if (bestBrain) {
        return bestBrain;
    }
    console.warn("No best brain recorded yet. Returning current Q-table if available.");
    return qTable; // Fallback to the current Q-table.
}

// Expose AI control functions to the global scope (via window object) for UI interaction.
window.ai = {
    startTraining,
    stopTraining,
    getBestBrain,
    // Optional: A function to set parameters if needed before training starts explicitly.
    setQLearningParameters: (lr, df, eps, decay) => {
        LEARNING_RATE = lr;
        DISCOUNT_FACTOR = df;
        EXPLORATION_RATE = eps;
        EPSILON_DECAY = decay;
    }
};

console.log("ai.js loaded with comments.");
