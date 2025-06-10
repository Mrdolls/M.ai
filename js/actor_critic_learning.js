/**
 * Actor-Critic Deep Reinforcement Learning
 * Implementation avancée combinant les avantages du Q-Learning et de l'apprentissage par imitation
 * Utilise deux réseaux : Actor (politique) et Critic (valeur)
 */
class ActorCriticLearning {
    constructor() {
        this.actorNetwork = null;
        this.criticNetwork = null;
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
        this.doubleQLearning = true;
        this.dueling = true;

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
            // Actor Network (Policy) - outputs action probabilities
            this.actorNetwork = new brain.NeuralNetwork({
                hiddenLayers: [256, 128, 64],
                learningRate: this.learningRate,
                activation: 'sigmoid',
                outputActivation: 'softmax' // For probability distribution
            });

            // Critic Network (Value function) - outputs state value
            this.criticNetwork = new brain.NeuralNetwork({
                hiddenLayers: [256, 128, 64],
                learningRate: this.learningRate,
                activation: 'relu'
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
        try {
            const newNetwork = new brain.NeuralNetwork({
                hiddenLayers: [256, 128, 64],
                learningRate: this.learningRate,
                activation: sourceNetwork === this.actorNetwork ? 'sigmoid' : 'relu'
            });

            if (sourceNetwork) {
                const json = sourceNetwork.toJSON();
                newNetwork.fromJSON(json);
            }

            return newNetwork;
        } catch (error) {
            console.error('Error cloning network:', error);
            return null;
        }
    }

    /**
     * Update parameters from UI
     */
    updateParameters() {
        this.learningRate = parseFloat(document.getElementById('actorCriticLearningRateInput')?.value) || 0.001;
        this.batchSize = parseInt(document.getElementById('actorCriticBatchSizeInput')?.value) || 64;
        this.gamma = parseFloat(document.getElementById('discountFactorInput')?.value) || 0.99;
        this.tau = parseFloat(document.getElementById('softUpdateInput')?.value) || 0.01;
        this.entropyCoefficient = parseFloat(document.getElementById('entropyCoeffInput')?.value) || 0.01;

        // Reinitialize networks with new parameters
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
        try {
            return {
                actor: this.actorNetwork.toJSON(),
                critic: this.criticNetwork.toJSON(),
                targetActor: this.targetActorNetwork.toJSON(),
                targetCritic: this.targetCriticNetwork.toJSON(),
                parameters: {
                    learningRate: this.learningRate,
                    gamma: this.gamma,
                    epsilon: this.epsilon,
                    tau: this.tau,
                    entropyCoefficient: this.entropyCoefficient
                },
                trainingHistory: this.trainingHistory,
                trainingEpisode: this.trainingEpisode
            };
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
            this.actorNetwork.fromJSON(modelData.actor);
            this.criticNetwork.fromJSON(modelData.critic);
            this.targetActorNetwork.fromJSON(modelData.targetActor);
            this.targetCriticNetwork.fromJSON(modelData.targetCritic);

            if (modelData.parameters) {
                Object.assign(this, modelData.parameters);
            }

            this.trainingHistory = modelData.trainingHistory || [];
            this.trainingEpisode = modelData.trainingEpisode || 0;

            console.log('Actor-Critic model loaded successfully');
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