/**
 * Actor-Critic Deep Reinforcement Learning
 * Implementation avancée combinant les avantages du Q-Learning et de l'apprentissage par imitation
 * Utilise deux réseaux : Actor (politique) et Critic (valeur)
 */
class ActorCriticLearning {
    constructor() {
        this.actorNetwork = null;
        this.criticNetwork = null;
        this.targetActorNetwork = null; // Added for completeness from initializeNetworks
        this.targetCriticNetwork = null; // Added for completeness from initializeNetworks
        this.isTraining = false;
        this.experience = [];
        this.maxExperienceSize = 10000;
        this.batchSize = 64;
        this.learningRate = 0.001;
        this.gamma = 0.99; // Discount factor
        this.epsilon = 1.0;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.01;
        this.tau = 0.01; // Soft update parameter
        this.advantageNormalization = true;
        this.entropyCoefficient = 0.01;

        // Advanced features
        this.prioritizedReplay = true;
        this.doubleQLearning = true; // Note: This is a concept, not just a parameter to set on network
        this.dueling = true; // Note: This is a concept/architecture, not just a parameter

        this.trainingEpisode = 0;
        this.trainingHistory = [];
        this.avgReward = 0;
        this.bestReward = -Infinity;

        this.initializeNetworks();
    }

    /**
     * Initialize Actor and Critic networks
     */
    initializeNetworks() {
        try {
            // Define network structures
            const inputSize = 20; // Based on stateToFeatures
            const actorOutputSize = 4; // Number of actions
            const criticOutputSize = 1; // State value

            // Actor Network (Policy) - outputs action probabilities
            this.actorNetwork = new brain.NeuralNetwork({
                inputSize: inputSize,
                hiddenLayers: [256, 128, 64],
                outputSize: actorOutputSize,
                learningRate: this.learningRate,
                activation: 'sigmoid', // Common for hidden layers in policy networks
                outputActivation: 'softmax' // For probability distribution
            });

            // Critic Network (Value function) - outputs state value
            this.criticNetwork = new brain.NeuralNetwork({
                inputSize: inputSize,
                hiddenLayers: [256, 128, 64],
                outputSize: criticOutputSize,
                learningRate: this.learningRate,
                activation: 'relu' // Common for value networks
            });

            // Target networks for stability (Double DQN)
            this.targetActorNetwork = this.cloneNetwork(this.actorNetwork);
            this.targetCriticNetwork = this.cloneNetwork(this.criticNetwork);

            console.log('Actor-Critic networks initialized successfully');
        } catch (error) {
            console.error('Error initializing Actor-Critic networks:', error);
        }
    }

    /**
     * Clone a neural network
     */
    cloneNetwork(sourceNetwork) {
        let sourceJson = null; // Define sourceJson outside try to be available in catch
        try {
            if (!sourceNetwork) {
                console.error('Source network is null, cannot clone.');
                return null;
            }

            sourceJson = sourceNetwork.toJSON(); // Get the full JSON of the source network

            // Determine if the source is an actor-like or critic-like network
            const isActorType = (sourceNetwork === this.actorNetwork || sourceNetwork === this.targetActorNetwork);

            // Prepare construction options, primarily deriving from sourceJson.options
            // Fallback to root properties of sourceJson if not in sourceJson.options
            const constructionOptions = {
                learningRate: this.learningRate, // Use current instance's LR
                activation: isActorType ? 'sigmoid' : 'relu',
                hiddenLayers: sourceJson.options && sourceJson.options.hiddenLayers ? sourceJson.options.hiddenLayers : sourceJson.hiddenLayers,
                inputSize: sourceJson.options && sourceJson.options.inputSize ? sourceJson.options.inputSize : sourceJson.inputSize,
                outputSize: sourceJson.options && sourceJson.options.outputSize ? sourceJson.options.outputSize : sourceJson.outputSize
            };

            if (isActorType) {
                constructionOptions.outputActivation = 'softmax';
            }

            // Log if crucial options are missing, as brain.js constructor might fail or create a differently structured network
            if (!constructionOptions.hiddenLayers || !constructionOptions.inputSize || !constructionOptions.outputSize) {
                console.warn('Cloning network: One or more structural parameters (hiddenLayers, inputSize, outputSize) are undefined. Constructor might rely on defaults or fromJSON to set them.', constructionOptions, sourceJson);
            }

            // Create a new network.
            // Pass only the necessary structural and activation options to the constructor.
            // fromJSON will handle the rest (weights, biases, and potentially overriding some options if they are part of the JSON structure it expects).
            const newNetwork = new brain.NeuralNetwork(constructionOptions);

            // Apply the full state from the source network's JSON
            // This will set weights, biases, and potentially other specific options stored in the JSON.
            newNetwork.fromJSON(sourceJson);

            return newNetwork;
        } catch (error) {
            // Log the error and the JSON that might have caused it for easier debugging
            console.error('Error cloning network:', error, sourceJson || 'sourceNetwork is null or sourceJson could not be generated');
            return null;
        }
    }

    /**
     * Update parameters from UI
     */
    updateParameters() {
        const lrElement = document.getElementById('actorCriticLearningRateInput');
        if (lrElement) {
            const parsedLR = parseFloat(lrElement.value);
            this.learningRate = !isNaN(parsedLR) ? parsedLR : this.learningRate;
        } else {
            console.warn("Element with ID 'actorCriticLearningRateInput' not found. Using current learning rate.");
        }

        const batchSizeElement = document.getElementById('actorCriticBatchSizeInput');
        if (batchSizeElement) {
            const parsedBS = parseInt(batchSizeElement.value);
            this.batchSize = !isNaN(parsedBS) ? parsedBS : this.batchSize;
        } else {
            console.warn("Element with ID 'actorCriticBatchSizeInput' not found. Using current batch size.");
        }

        const gammaElement = document.getElementById('actorCriticDiscountFactorInput'); // Updated ID
        if (gammaElement) {
            const parsedGamma = parseFloat(gammaElement.value);
            this.gamma = !isNaN(parsedGamma) ? parsedGamma : this.gamma;
        } else {
            console.warn("Element with ID 'actorCriticDiscountFactorInput' not found. Using current gamma.");
        }

        const tauElement = document.getElementById('softUpdateInput');
        if (tauElement) {
            const parsedTau = parseFloat(tauElement.value);
            this.tau = !isNaN(parsedTau) ? parsedTau : this.tau;
        } else {
            console.warn("Element with ID 'softUpdateInput' not found. Using current tau.");
        }

        const entropyCoeffElement = document.getElementById('entropyCoeffInput');
        if (entropyCoeffElement) {
            const parsedEntropy = parseFloat(entropyCoeffElement.value);
            this.entropyCoefficient = !isNaN(parsedEntropy) ? parsedEntropy : this.entropyCoefficient;
        } else {
            console.warn("Element with ID 'entropyCoeffInput' not found. Using current entropy coefficient.");
        }

        // Reinitialize networks with new parameters only if any relevant parameter changed.
        // For simplicity here, we still reinitialize. A more complex check could compare old vs new values.
        this.initializeNetworks();
    }

    /**
     * Convert game state to feature vector
     */
    stateToFeatures(gameState) {
        const features = [];

        // Agent position (normalized)
        features.push(gameState.agentX / gameState.gameWidth);
        features.push(gameState.agentY / gameState.gameHeight);

        // Agent velocity
        features.push(gameState.velocityX || 0);
        features.push(gameState.velocityY || 0);

        // Closest circle information
        if (gameState.circles && gameState.circles.length > 0) {
            const closest = this.findClosestCircle(gameState);
            features.push(closest.x / gameState.gameWidth);
            features.push(closest.y / gameState.gameHeight);
            features.push(closest.distance / Math.sqrt(gameState.gameWidth ** 2 + gameState.gameHeight ** 2));
            features.push(closest.angle / (2 * Math.PI));

            // Distance to each circle (up to 5)
            const sortedCircles = gameState.circles
                .map(circle => this.calculateDistance(gameState.agentX, gameState.agentY, circle.x, circle.y))
                .sort((a, b) => a - b)
                .slice(0, 5);

            while (sortedCircles.length < 5) sortedCircles.push(1.0); // Max distance
            features.push(...sortedCircles.map(d => d / Math.sqrt(gameState.gameWidth ** 2 + gameState.gameHeight ** 2)));
        } else {
            features.push(0.5, 0.5, 1.0, 0.0);
            features.push(1.0, 1.0, 1.0, 1.0, 1.0); // No circles
        }

        // Wall distances (normalized)
        features.push(gameState.agentX / gameState.gameWidth);
        features.push((gameState.gameWidth - gameState.agentX) / gameState.gameWidth);
        features.push(gameState.agentY / gameState.gameHeight);
        features.push((gameState.gameHeight - gameState.agentY) / gameState.gameHeight);

        // Game context
        features.push((gameState.circles?.length || 0) / (gameState.initialCircles || 5));
        features.push(gameState.timeRemaining ? gameState.timeRemaining / gameState.maxTime : 0.5);
        features.push(gameState.score / 100); // Normalized score

        return features;
    }

    /**
     * Get action from actor network with exploration
     */
    getAction(gameState, training = true) {
        try {
            const features = this.stateToFeatures(gameState);

            // Get action probabilities from actor network
            const actionProbs = this.actorNetwork.run(features);

            if (training && Math.random() < this.epsilon) {
                // Exploration: random action
                return Math.floor(Math.random() * 4);
            } else {
                // Exploitation: choose action based on probabilities
                if (training) {
                    return this.sampleFromDistribution(actionProbs);
                } else {
                    return actionProbs.indexOf(Math.max(...actionProbs));
                }
            }
        } catch (error) {
            console.error('Error getting action:', error);
            return Math.floor(Math.random() * 4);
        }
    }

    /**
     * Sample action from probability distribution
     */
    sampleFromDistribution(probabilities) {
        const rand = Math.random();
        let cumulative = 0;

        for (let i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (rand < cumulative) {
                return i;
            }
        }

        return probabilities.length - 1;
    }

    /**
     * Store experience in replay buffer
     */
    storeExperience(state, action, reward, nextState, done) {
        const experience = {
            state: this.stateToFeatures(state),
            action: action,
            reward: reward,
            nextState: this.stateToFeatures(nextState),
            done: done,
            priority: 1.0, // For prioritized replay
            timestamp: Date.now()
        };

        this.experience.push(experience);

        // Remove old experiences if buffer is full
        if (this.experience.length > this.maxExperienceSize) {
            this.experience.shift();
        }
    }

    /**
     * Train the Actor-Critic networks
     */
    async trainStep() {
        if (this.experience.length < this.batchSize) {
            return false;
        }

        try {
            // Sample batch
            const batch = this.sampleBatch();

            // Prepare training data
            const actorTrainingData = [];
            const criticTrainingData = [];

            for (const exp of batch) {
                // Calculate advantage
                const stateValue = this.criticNetwork.run(exp.state)[0] || 0;
                const nextStateValue = exp.done ? 0 : (this.criticNetwork.run(exp.nextState)[0] || 0);
                const tdTarget = exp.reward + this.gamma * nextStateValue;
                const advantage = tdTarget - stateValue;

                // Actor training data (policy gradient with advantage)
                const actionProbs = this.actorNetwork.run(exp.state);
                const targetActionProbs = [...actionProbs];

                // Apply advantage to selected action
                if (this.advantageNormalization) {
                    const normalizedAdvantage = advantage / (Math.abs(advantage) + 1e-8);
                    targetActionProbs[exp.action] += this.learningRate * normalizedAdvantage;
                } else {
                    targetActionProbs[exp.action] += this.learningRate * advantage;
                }

                // Add entropy bonus for exploration
                const entropy = this.calculateEntropy(actionProbs);
                for (let i = 0; i < targetActionProbs.length; i++) {
                    targetActionProbs[i] += this.entropyCoefficient * entropy;
                }

                // Normalize probabilities
                const sum = targetActionProbs.reduce((a, b) => a + b, 0);
                for (let i = 0; i < targetActionProbs.length; i++) {
                    targetActionProbs[i] = Math.max(0.001, targetActionProbs[i] / sum);
                }

                actorTrainingData.push({
                    input: exp.state,
                    output: targetActionProbs
                });

                // Critic training data (value function)
                criticTrainingData.push({
                    input: exp.state,
                    output: [tdTarget]
                });
            }

            // Train networks
            await this.trainNetworks(actorTrainingData, criticTrainingData);

            // Update target networks (soft update)
            this.updateTargetNetworks();

            // Decay epsilon
            this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);

            return true;
        } catch (error) {
            console.error('Error in training step:', error);
            return false;
        }
    }

    /**
     * Sample batch from experience buffer
     */
    sampleBatch() {
        if (this.prioritizedReplay) {
            return this.prioritizedSample();
        } else {
            // Random sampling
            const batch = [];
            for (let i = 0; i < this.batchSize; i++) {
                const index = Math.floor(Math.random() * this.experience.length);
                batch.push(this.experience[index]);
            }
            return batch;
        }
    }

    /**
     * Prioritized experience replay sampling
     */
    prioritizedSample() {
        const priorities = this.experience.map(exp => exp.priority);
        const totalPriority = priorities.reduce((sum, p) => sum + p, 0);

        const batch = [];
        for (let i = 0; i < Math.min(this.batchSize, this.experience.length); i++) {
            const rand = Math.random() * totalPriority;
            let cumulative = 0;

            for (let j = 0; j < this.experience.length; j++) {
                cumulative += priorities[j];
                if (rand < cumulative) {
                    batch.push(this.experience[j]);
                    break;
                }
            }
        }

        return batch;
    }

    /**
     * Train actor and critic networks
     */
    async trainNetworks(actorData, criticData) {
        return new Promise((resolve) => {
            try {
                // Train critic network
                this.criticNetwork.train(criticData, {
                    iterations: 1,
                    learningRate: this.learningRate,
                    log: false
                });

                // Train actor network
                this.actorNetwork.train(actorData, {
                    iterations: 1,
                    learningRate: this.learningRate,
                    log: false
                });

                resolve(true);
            } catch (error) {
                console.error('Error training networks:', error);
                resolve(false);
            }
        });
    }

    /**
     * Update target networks with soft update
     */
    updateTargetNetworks() {
        try {
            // Soft update: target = tau * current + (1 - tau) * target
            this.softUpdateNetwork(this.actorNetwork, this.targetActorNetwork);
            this.softUpdateNetwork(this.criticNetwork, this.targetCriticNetwork);
        } catch (error) {
            console.error('Error updating target networks:', error);
        }
    }

    /**
     * Soft update network weights
     */
    softUpdateNetwork(sourceNetwork, targetNetwork) {
        // This is a simplified implementation
        // In practice, you would update weights layer by layer
        if (Math.random() < this.tau) {
            const sourceJson = sourceNetwork.toJSON();
            targetNetwork.fromJSON(sourceJson);
        }
    }

    /**
     * Calculate entropy of action probabilities
     */
    calculateEntropy(probabilities) {
        let entropy = 0;
        for (const prob of probabilities) {
            if (prob > 0) {
                entropy -= prob * Math.log(prob);
            }
        }
        return entropy;
    }

    /**
     * Find closest circle to agent
     */
    findClosestCircle(gameState) {
        if (!gameState.circles || gameState.circles.length === 0) {
            return { x: gameState.gameWidth / 2, y: gameState.gameHeight / 2, distance: 0, angle: 0 };
        }

        let closest = null;
        let minDistance = Infinity;

        gameState.circles.forEach(circle => {
            const distance = this.calculateDistance(gameState.agentX, gameState.agentY, circle.x, circle.y);
            if (distance < minDistance) {
                minDistance = distance;
                closest = {
                    x: circle.x,
                    y: circle.y,
                    distance: distance,
                    angle: Math.atan2(circle.y - gameState.agentY, circle.x - gameState.agentX)
                };
            }
        });

        return closest || { x: gameState.gameWidth / 2, y: gameState.gameHeight / 2, distance: 0, angle: 0 };
    }

    /**
     * Calculate distance between two points
     */
    calculateDistance(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Convert action index to action name
     */
    indexToAction(actionIndex) {
        const actions = ['up', 'down', 'left', 'right'];
        return actions[actionIndex] || null;
    }

    /**
     * Predict action for given game state
     */
    predict(gameState) {
        try {
            const actionIndex = this.getAction(gameState, false);
            return this.indexToAction(actionIndex);
        } catch (error) {
            console.error('Error in prediction:', error);
            return null;
        }
    }

    /**
     * Update training progress
     */
    updateTrainingProgress(episode, avgReward, epsilon) {
        this.trainingEpisode = episode;
        this.avgReward = avgReward;
        this.epsilon = epsilon;

        // Update UI
        document.getElementById('statusMessage').textContent =
            `Entraînement Actor-Critic - Épisode ${episode}`;

        // Update epsilon display
        document.getElementById('epsilonDisplay').textContent = epsilon.toFixed(3);

        console.log(`Episode ${episode}: Avg Reward = ${avgReward.toFixed(2)}, Epsilon = ${epsilon.toFixed(3)}`);
    }

    /**
     * Evaluate model performance
     */
    evaluate(testEpisodes = 10) {
        let totalReward = 0;
        let successfulEpisodes = 0;

        // This would need integration with the game environment
        // For now, return estimated performance based on training history
        if (this.trainingHistory.length > 0) {
            const recentHistory = this.trainingHistory.slice(-10);
            totalReward = recentHistory.reduce((sum, h) => sum + h.reward, 0);
            successfulEpisodes = recentHistory.filter(h => h.reward > 0).length;
        }

        return {
            avgReward: totalReward / Math.max(testEpisodes, 1),
            successRate: successfulEpisodes / Math.max(testEpisodes, 1),
            convergence: this.trainingEpisode
        };
    }

    /**
     * Save model to JSON
     */
    saveModel() {
        const modelVersion = "1.0.0";
        try {
            const modelData = {
                modelType: "ActorCriticLearning",
                version: modelVersion,
                timestamp: Date.now(),
                parameters: {
                    learningRate: this.learningRate,
                    gamma: this.gamma,
                    epsilon: this.epsilon,
                    epsilonDecay: this.epsilonDecay,
                    epsilonMin: this.epsilonMin,
                    tau: this.tau,
                    entropyCoefficient: this.entropyCoefficient,
                    maxExperienceSize: this.maxExperienceSize,
                    batchSize: this.batchSize,
                    advantageNormalization: this.advantageNormalization,
                    prioritizedReplay: this.prioritizedReplay,
                    doubleQLearning: this.doubleQLearning, // conceptual, but saved for info
                    dueling: this.dueling, // conceptual, but saved for info
                },
                networks: {
                    actor: this.actorNetwork ? this.actorNetwork.toJSON() : null,
                    critic: this.criticNetwork ? this.criticNetwork.toJSON() : null,
                    targetActor: this.targetActorNetwork ? this.targetActorNetwork.toJSON() : null,
                    targetCritic: this.targetCriticNetwork ? this.targetCriticNetwork.toJSON() : null,
                },
                trainingState: {
                    trainingHistory: this.trainingHistory,
                    trainingEpisode: this.trainingEpisode,
                },
                experienceBuffer: JSON.stringify(this.experience) // Serialize experience buffer
            };
            console.log('Actor-Critic model saved successfully.');
            return modelData;
        } catch (error) {
            console.error('Error saving Actor-Critic model:', error);
            return null;
        }
    }

    /**
     * Load model from JSON
     */
    loadModel(modelData) {
        try {
            // Handle new "BrainOnly" model type
            if (modelData && modelData.modelType === "ActorCriticLearning_BrainOnly") {
                console.log('Loading Actor-Critic model (BrainOnly format)...');

                // Ensure actorNetwork exists and is a brain.NeuralNetwork instance
                if (!this.actorNetwork) {
                    this.actorNetwork = new brain.NeuralNetwork(); // fromJSON will structure it
                } else if (!(this.actorNetwork instanceof brain.NeuralNetwork)) {
                    console.warn('this.actorNetwork was not a brain.NeuralNetwork instance. Re-initializing.');
                    this.actorNetwork = new brain.NeuralNetwork();
                }

                // Ensure criticNetwork exists and is a brain.NeuralNetwork instance
                if (!this.criticNetwork) {
                    this.criticNetwork = new brain.NeuralNetwork(); // fromJSON will structure it
                } else if (!(this.criticNetwork instanceof brain.NeuralNetwork)) {
                    console.warn('this.criticNetwork was not a brain.NeuralNetwork instance. Re-initializing.');
                    this.criticNetwork = new brain.NeuralNetwork();
                }

                let actorLoaded = false;
                if (modelData.networks && modelData.networks.actor) {
                    this.actorNetwork.fromJSON(modelData.networks.actor);
                    actorLoaded = true;
                    console.log('Actor network loaded from BrainOnly model.');
                } else {
                    console.warn('No actor network data found in BrainOnly model.');
                }

                let criticLoaded = false;
                if (modelData.networks && modelData.networks.critic) {
                    this.criticNetwork.fromJSON(modelData.networks.critic);
                    criticLoaded = true;
                    console.log('Critic network loaded from BrainOnly model.');
                } else {
                    console.warn('No critic network data found in BrainOnly model.');
                }

                // Optionally, re-initialize target networks if they are fundamental to operation after any load
                // However, for "BrainOnly", the assumption is we might not need them or they are not provided.
                // If they are needed, they should be cloned from the newly loaded actor/critic.
                if (this.actorNetwork && actorLoaded) { // If actor was loaded
                    this.targetActorNetwork = this.cloneNetwork(this.actorNetwork);
                }
                if (this.criticNetwork && criticLoaded) { // If critic was loaded
                    this.targetCriticNetwork = this.cloneNetwork(this.criticNetwork);
                }

                console.log('Actor-Critic (BrainOnly) model loaded successfully.');
                return true;
            }

            // Existing logic for the old format "ActorCriticLearning"
            if (!modelData || !modelData.modelType || modelData.modelType !== "ActorCriticLearning") {
                console.error('Invalid or incompatible model data format for ActorCriticLearning (old format).');
                return false;
            }

            console.log(`Loading Actor-Critic model version: ${modelData.version || 'unknown'} (old format)`);
            if (modelData.version !== "1.0.0") {
                console.warn(`Attempting to load model version ${modelData.version}, current version is 1.0.0. Compatibility issues may arise.`);
            }

            // Restore parameters
            if (modelData.parameters) {
                const params = modelData.parameters;
                this.learningRate = params.learningRate !== undefined ? params.learningRate : this.learningRate;
                this.gamma = params.gamma !== undefined ? params.gamma : this.gamma;
                this.epsilon = params.epsilon !== undefined ? params.epsilon : this.epsilon;
                this.epsilonDecay = params.epsilonDecay !== undefined ? params.epsilonDecay : this.epsilonDecay;
                this.epsilonMin = params.epsilonMin !== undefined ? params.epsilonMin : this.epsilonMin;
                this.tau = params.tau !== undefined ? params.tau : this.tau;
                this.entropyCoefficient = params.entropyCoefficient !== undefined ? params.entropyCoefficient : this.entropyCoefficient;
                this.maxExperienceSize = params.maxExperienceSize !== undefined ? params.maxExperienceSize : this.maxExperienceSize;
                this.batchSize = params.batchSize !== undefined ? params.batchSize : this.batchSize;
                this.advantageNormalization = params.advantageNormalization !== undefined ? params.advantageNormalization : this.advantageNormalization;
                this.prioritizedReplay = params.prioritizedReplay !== undefined ? params.prioritizedReplay : this.prioritizedReplay;
                this.doubleQLearning = params.doubleQLearning !== undefined ? params.doubleQLearning : this.doubleQLearning;
                this.dueling = params.dueling !== undefined ? params.dueling : this.dueling;
            }

            // Initialize networks if they don't exist, then load from JSON
            if (!this.actorNetwork) this.initializeNetworks(); // This will init all four

            if (modelData.networks) {
                if (modelData.networks.actor) {
                    this.actorNetwork.fromJSON(modelData.networks.actor);
                } else {
                    console.warn("No actor network data found in saved model. It remains initialized.");
                }
                if (modelData.networks.critic) {
                    this.criticNetwork.fromJSON(modelData.networks.critic);
                } else {
                    console.warn("No critic network data found in saved model. It remains initialized.");
                }
                if (modelData.networks.targetActor) {
                    this.targetActorNetwork.fromJSON(modelData.networks.targetActor);
                } else {
                    console.warn("No target actor network data found in saved model. It remains initialized.");
                }
                if (modelData.networks.targetCritic) {
                    this.targetCriticNetwork.fromJSON(modelData.networks.targetCritic);
                } else {
                    console.warn("No target critic network data found in saved model. It remains initialized.");
                }
            } else {
                console.warn("No network data found in saved model. Networks remain initialized.");
            }

            // Restore training state
            if (modelData.trainingState) {
                this.trainingHistory = modelData.trainingState.trainingHistory || [];
                this.trainingEpisode = modelData.trainingState.trainingEpisode || 0;
            }

            // Restore experience buffer
            if (modelData.experienceBuffer) {
                try {
                    this.experience = JSON.parse(modelData.experienceBuffer);
                    // Ensure experience array doesn't exceed max size after loading
                    if (this.experience.length > this.maxExperienceSize) {
                        this.experience = this.experience.slice(this.experience.length - this.maxExperienceSize);
                    }
                } catch (e) {
                    console.error("Error parsing experience buffer:", e);
                    this.experience = [];
                }
            } else {
                this.experience = []; // Initialize if not present
            }

            // It might be good to re-initialize networks if learningRate changed
            // For now, brain.js networks usually handle learning rate internally or it's passed during train
            // this.initializeNetworks(); // if parameters like learningRate that affect network structure/init are changed

            console.log('Actor-Critic model loaded successfully.');
            return true;
        } catch (error) {
            console.error('Error loading Actor-Critic model:', error);
            return false;
        }
    }

    /**
     * Reset the learning system
     */
    reset() {
        this.experience = [];
        this.trainingHistory = [];
        this.trainingEpisode = 0;
        this.epsilon = 1.0;
        this.avgReward = 0;
        this.bestReward = -Infinity;
        this.initializeNetworks();
        console.log('Actor-Critic system reset');
    }

    /**
     * Get training statistics
     */
    getTrainingStats() {
        return {
            episode: this.trainingEpisode,
            avgReward: this.avgReward,
            bestReward: this.bestReward,
            epsilon: this.epsilon,
            experienceSize: this.experience.length,
            convergence: this.trainingHistory.length > 100 ?
                this.trainingHistory.slice(-100).findIndex(h => h.reward > this.avgReward * 0.9) : -1
        };
    }
}
