/**
 * Neural Network utilities and management for the multi-modal trainer
 */
class NeuralNetworkManager {
    constructor() {
        this.networks = {
            genetic: [],
            qlearning: null,
            imitation: null
        };
        this.populationSize = 50;
        this.inputSize = 12;
        this.outputSize = 4;
        this.hiddenLayers = [64, 32];
    }

    /**
     * Create a new empty neural network
     */
    createNetwork(options = {}) {
        const config = {
            inputSize: options.inputSize || this.inputSize,
            hiddenLayers: options.hiddenLayers || this.hiddenLayers,
            outputSize: options.outputSize || this.outputSize,
            activation: options.activation || 'sigmoid',
            learningRate: options.learningRate || 0.1
        };

        try {
            return new brain.NeuralNetwork({
                hiddenLayers: config.hiddenLayers,
                activation: config.activation,
                learningRate: config.learningRate
            });
        } catch (error) {
            console.error('Error creating neural network:', error);
            return null;
        }
    }

    /**
     * Initialize population of networks for genetic algorithm
     */
    initializeGeneticPopulation(populationSize = null) {
        this.populationSize = populationSize || this.populationSize;
        this.networks.genetic = [];

        for (let i = 0; i < this.populationSize; i++) {
            const network = this.createNetwork();
            if (network) {
                // Initialize with random weights
                this.randomizeNetworkWeights(network);
                this.networks.genetic.push({
                    network: network,
                    fitness: 0,
                    id: i
                });
            }
        }

        console.log(`Initialized genetic population with ${this.networks.genetic.length} networks`);
        return this.networks.genetic;
    }

    /**
     * Initialize Q-Learning network
     */
    initializeQLearningNetwork(config = {}) {
        this.networks.qlearning = this.createNetwork({
            hiddenLayers: config.hiddenLayers || [128, 64],
            learningRate: config.learningRate || 0.01,
            activation: 'relu'
        });

        if (this.networks.qlearning) {
            console.log('Q-Learning network initialized');
        }

        return this.networks.qlearning;
    }

    /**
     * Initialize Imitation Learning network
     */
    initializeImitationNetwork(config = {}) {
        this.networks.imitation = this.createNetwork({
            hiddenLayers: config.hiddenLayers || [128, 64, 32],
            learningRate: config.learningRate || 0.01,
            activation: 'sigmoid'
        });

        if (this.networks.imitation) {
            console.log('Imitation learning network initialized');
        }

        return this.networks.imitation;
    }

    /**
     * Randomize network weights for genetic algorithm initialization
     */
    randomizeNetworkWeights(network) {
        if (!network) return;

        try {
            // Generate random training data to initialize weights
            const dummyData = [];
            for (let i = 0; i < 10; i++) {
                const input = Array(this.inputSize).fill(0).map(() => Math.random());
                const output = Array(this.outputSize).fill(0).map(() => Math.random());
                dummyData.push({ input, output });
            }

            // Train for just 1 iteration to set random weights
            network.train(dummyData, { iterations: 1, log: false });
        } catch (error) {
            console.error('Error randomizing network weights:', error);
        }
    }

    /**
     * Clone a neural network
     */
    cloneNetwork(sourceNetwork) {
        if (!sourceNetwork) return null;

        try {
            const newNetwork = this.createNetwork();
            const json = sourceNetwork.toJSON();
            newNetwork.fromJSON(json);
            return newNetwork;
        } catch (error) {
            console.error('Error cloning network:', error);
            return null;
        }
    }

    /**
     * Mutate network weights for genetic algorithm
     */
    mutateNetwork(network, mutationRate = 0.1, mutationStrength = 0.5) {
        if (!network) return network;

        try {
            const json = network.toJSON();

            // Mutate weights in the JSON representation
            if (json.layers) {
                json.layers.forEach(layer => {
                    if (layer.weights) {
                        layer.weights = layer.weights.map(weight => {
                            if (Math.random() < mutationRate) {
                                return weight + (Math.random() - 0.5) * mutationStrength;
                            }
                            return weight;
                        });
                    }
                    if (layer.biases) {
                        layer.biases = layer.biases.map(bias => {
                            if (Math.random() < mutationRate) {
                                return bias + (Math.random() - 0.5) * mutationStrength;
                            }
                            return bias;
                        });
                    }
                });
            }

            // Create new network from mutated JSON
            const mutatedNetwork = this.createNetwork();
            mutatedNetwork.fromJSON(json);
            return mutatedNetwork;
        } catch (error) {
            console.error('Error mutating network:', error);
            return network;
        }
    }

    /**
     * Crossover between two networks for genetic algorithm
     */
    crossoverNetworks(parent1, parent2, crossoverRate = 0.8) {
        if (!parent1 || !parent2) return null;

        try {
            const json1 = parent1.toJSON();
            const json2 = parent2.toJSON();

            // Create child by mixing weights from both parents
            const childJson = JSON.parse(JSON.stringify(json1));

            if (json1.layers && json2.layers) {
                for (let i = 0; i < childJson.layers.length; i++) {
                    if (childJson.layers[i].weights && json2.layers[i].weights) {
                        childJson.layers[i].weights = childJson.layers[i].weights.map((weight, idx) => {
                            if (Math.random() < crossoverRate && json2.layers[i].weights[idx] !== undefined) {
                                return json2.layers[i].weights[idx];
                            }
                            return weight;
                        });
                    }

                    if (childJson.layers[i].biases && json2.layers[i].biases) {
                        childJson.layers[i].biases = childJson.layers[i].biases.map((bias, idx) => {
                            if (Math.random() < crossoverRate && json2.layers[i].biases[idx] !== undefined) {
                                return json2.layers[i].biases[idx];
                            }
                            return bias;
                        });
                    }
                }
            }

            const childNetwork = this.createNetwork();
            childNetwork.fromJSON(childJson);
            return childNetwork;
        } catch (error) {
            console.error('Error in crossover:', error);
            return this.cloneNetwork(parent1);
        }
    }

    /**
     * Evaluate network performance
     */
    evaluateNetwork(network, gameState) {
        if (!network || !gameState) return 0;

        try {
            const input = this.gameStateToInput(gameState);
            const output = network.run(input);
            return this.calculateFitness(output, gameState);
        } catch (error) {
            console.error('Error evaluating network:', error);
            return 0;
        }
    }

    /**
     * Convert game state to neural network input
     */
    gameStateToInput(gameState) {
        const input = [];

        // Normalize agent position
        input.push(gameState.agentX / gameState.gameWidth);
        input.push(gameState.agentY / gameState.gameHeight);

        // Agent velocity
        input.push(gameState.velocityX || 0);
        input.push(gameState.velocityY || 0);

        // Closest circle information
        const closest = this.findClosestTarget(gameState);
        input.push(closest.x / gameState.gameWidth);
        input.push(closest.y / gameState.gameHeight);
        input.push(closest.distance / Math.sqrt(gameState.gameWidth ** 2 + gameState.gameHeight ** 2));
        input.push(closest.angle / (2 * Math.PI));

        // Wall distances
        input.push(gameState.agentX / gameState.gameWidth);
        input.push((gameState.gameWidth - gameState.agentX) / gameState.gameWidth);
        input.push(gameState.agentY / gameState.gameHeight);
        input.push((gameState.gameHeight - gameState.agentY) / gameState.gameHeight);

        return input;
    }

    /**
     * Find closest target (circle) to agent
     */
    findClosestTarget(gameState) {
        if (!gameState.circles || gameState.circles.length === 0) {
            return {
                x: gameState.gameWidth / 2,
                y: gameState.gameHeight / 2,
                distance: 0,
                angle: 0
            };
        }

        let closest = gameState.circles[0];
        let minDistance = Infinity;

        gameState.circles.forEach(circle => {
            const dx = circle.x - gameState.agentX;
            const dy = circle.y - gameState.agentY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < minDistance) {
                minDistance = distance;
                closest = circle;
            }
        });

        const dx = closest.x - gameState.agentX;
        const dy = closest.y - gameState.agentY;

        return {
            x: closest.x,
            y: closest.y,
            distance: minDistance,
            angle: Math.atan2(dy, dx)
        };
    }

    /**
     * Calculate fitness score for genetic algorithm
     */
    calculateFitness(output, gameState) {
        let fitness = 0;

        // Reward for having clear action preferences
        const maxOutput = Math.max(...output);
        if (maxOutput > 0.7) {
            fitness += 1;
        }

        // Penalize indecisive outputs
        const variance = this.calculateVariance(output);
        fitness += variance * 5;

        // Reward for logical movement towards targets
        const closest = this.findClosestTarget(gameState);
        const actionIndex = output.indexOf(maxOutput);

        if (this.isActionTowardsTarget(actionIndex, closest, gameState)) {
            fitness += 10;
        }

        return fitness;
    }

    /**
     * Check if action is towards target
     */
    isActionTowardsTarget(actionIndex, target, gameState) {
        const dx = target.x - gameState.agentX;
        const dy = target.y - gameState.agentY;

        // Actions: [up, down, left, right]
        switch (actionIndex) {
            case 0: return dy < 0; // Up
            case 1: return dy > 0; // Down
            case 2: return dx < 0; // Left
            case 3: return dx > 0; // Right
            default: return false;
        }
    }

    /**
     * Calculate variance of array
     */
    calculateVariance(arr) {
        const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
        const variance = arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / arr.length;
        return variance;
    }

    /**
     * Get network output as action
     */
    getNetworkAction(network, gameState) {
        if (!network || !gameState) return null;

        try {
            const input = this.gameStateToInput(gameState);
            const output = network.run(input);

            // Find the action with highest activation
            const maxIndex = output.indexOf(Math.max(...output));
            const actions = ['up', 'down', 'left', 'right'];

            return output[maxIndex] > 0.5 ? actions[maxIndex] : null;
        } catch (error) {
            console.error('Error getting network action:', error);
            return null;
        }
    }

    /**
     * Save network to JSON format
     */
    saveNetwork(network, metadata = {}) {
        if (!network) return null;

        try {
            return {
                network: network.toJSON(),
                metadata: {
                    timestamp: Date.now(),
                    type: metadata.type || 'unknown',
                    fitness: metadata.fitness || 0,
                    generation: metadata.generation || 0,
                    ...metadata
                }
            };
        } catch (error) {
            console.error('Error saving network:', error);
            return null;
        }
    }

    /**
     * Load network from JSON format
     */
    loadNetwork(data) {
        if (!data || !data.network) return null;

        try {
            const network = this.createNetwork();
            network.fromJSON(data.network);
            return {
                network: network,
                metadata: data.metadata || {}
            };
        } catch (error) {
            console.error('Error loading network:', error);
            return null;
        }
    }

    /**
     * Get network statistics
     */
    getNetworkStats(network) {
        if (!network) return null;

        try {
            const json = network.toJSON();
            let totalWeights = 0;
            let totalBiases = 0;

            if (json.layers) {
                json.layers.forEach(layer => {
                    if (layer.weights) totalWeights += layer.weights.length;
                    if (layer.biases) totalBiases += layer.biases.length;
                });
            }

            return {
                totalParameters: totalWeights + totalBiases,
                totalWeights: totalWeights,
                totalBiases: totalBiases,
                layers: json.layers ? json.layers.length : 0
            };
        } catch (error) {
            console.error('Error getting network stats:', error);
            return null;
        }
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.networks.genetic = [];
        this.networks.qlearning = null;
        this.networks.imitation = null;
    }
}
