/**
 * Imitation Learning Module
 * Implements Behavioral Cloning for learning from expert demonstrations
 */
class ImitationLearning {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.trainingHistory = [];
        this.currentEpoch = 0;
        this.maxEpochs = 100;
        this.learningRate = 0.01;
        this.batchSize = 32;
        this.validationSplit = 0.2;
        this.hiddenLayers = [128, 64, 32];
        this.noiseFactor = 0.1;

        this.initializeModel();
    }

    /**
     * Initialize the neural network model for behavioral cloning
     */
    initializeModel() {
        try {
            // Create neural network with Brain.js
            this.model = new brain.NeuralNetwork({
                hiddenLayers: this.hiddenLayers,
                learningRate: this.learningRate,
                activation: "sigmoid",
                iterations: 1, // We'll handle iterations manually
            });
        } catch (error) {
            console.error(
                "Error initializing imitation learning model:",
                error,
            );
        }
    }

    /**
     * Update model parameters from UI
     */
    updateParameters() {
        this.learningRate =
            parseFloat(
                document.getElementById("imitationLearningRateInput").value,
            ) || 0.01;
        this.batchSize =
            parseInt(document.getElementById("batchSizeInput").value) || 32;
        this.maxEpochs =
            parseInt(document.getElementById("epochsInput").value) || 100;
        this.noiseFactor =
            parseFloat(document.getElementById("noiseFactorInput").value) ||
            0.1;

        // Parse hidden layers
        const hiddenLayersStr =
            document.getElementById("hiddenLayersInput").value || "128,64,32";
        this.hiddenLayers = hiddenLayersStr
            .split(",")
            .map((layer) => parseInt(layer.trim()))
            .filter((n) => !isNaN(n));

        // Reinitialize model with new parameters
        this.initializeModel();
    }

    /**
     * Convert game state to feature vector for neural network input
     */
    stateToFeatures(gameState) {
        const features = [];

        // Agent position (normalized)
        features.push(gameState.agentX / gameState.gameWidth);
        features.push(gameState.agentY / gameState.gameHeight);

        // Agent velocity (if available)
        features.push(gameState.velocityX || 0);
        features.push(gameState.velocityY || 0);

        // Closest circle information
        if (gameState.circles && gameState.circles.length > 0) {
            const closest = this.findClosestCircle(gameState);
            features.push(closest.x / gameState.gameWidth);
            features.push(closest.y / gameState.gameHeight);
            features.push(
                closest.distance /
                    Math.sqrt(
                        gameState.gameWidth * gameState.gameWidth +
                            gameState.gameHeight * gameState.gameHeight,
                    ),
            );
            features.push(closest.angle / (2 * Math.PI));
        } else {
            features.push(0.5, 0.5, 1.0, 0.0); // Default values when no circles
        }

        // Wall distances (normalized)
        features.push(gameState.agentX / gameState.gameWidth); // Distance to left wall
        features.push(
            (gameState.gameWidth - gameState.agentX) / gameState.gameWidth,
        ); // Distance to right wall
        features.push(gameState.agentY / gameState.gameHeight); // Distance to top wall
        features.push(
            (gameState.gameHeight - gameState.agentY) / gameState.gameHeight,
        ); // Distance to bottom wall

        // Number of circles remaining (normalized)
        features.push(
            (gameState.circles ? gameState.circles.length : 0) /
                (gameState.initialCircles || 5),
        );

        // Time factor (if available)
        features.push(
            gameState.timeRemaining
                ? gameState.timeRemaining / gameState.maxTime
                : 0.5,
        );

        return features;
    }

    /**
     * Convert action to output format for neural network
     */
    actionToOutput(action) {
        const output = [0, 0, 0, 0]; // [up, down, left, right]

        switch (action.toLowerCase()) {
            case "arrowup":
            case "w":
            case "z": // French keyboard
                output[0] = 1;
                break;
            case "arrowdown":
            case "s":
                output[1] = 1;
                break;
            case "arrowleft":
            case "a":
            case "q": // French keyboard
                output[2] = 1;
                break;
            case "arrowright":
            case "d":
                output[3] = 1;
                break;
        }

        return output;
    }

    /**
     * Convert neural network output to action
     */
    outputToAction(output) {
        if (!output || output.length !== 4) return null;

        const maxIndex = output.indexOf(Math.max(...output));
        const actions = ["up", "down", "left", "right"];

        // Only return action if confidence is above threshold
        if (output[maxIndex] > 0.5) {
            return actions[maxIndex];
        }

        return null; // No action
    }

    /**
     * Find the closest circle to the agent
     */
    findClosestCircle(gameState) {
        if (!gameState.circles || gameState.circles.length === 0) {
            return {
                x: gameState.gameWidth / 2,
                y: gameState.gameHeight / 2,
                distance: 0,
                angle: 0,
            };
        }

        let closest = null;
        let minDistance = Infinity;

        gameState.circles.forEach((circle) => {
            const dx = circle.x - gameState.agentX;
            const dy = circle.y - gameState.agentY;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < minDistance) {
                minDistance = distance;
                closest = {
                    x: circle.x,
                    y: circle.y,
                    distance: distance,
                    angle: Math.atan2(dy, dx),
                };
            }
        });

        return (
            closest || {
                x: gameState.gameWidth / 2,
                y: gameState.gameHeight / 2,
                distance: 0,
                angle: 0,
            }
        );
    }

    /**
     * Prepare training data from demonstrations
     */
    prepareTrainingData(demonstrations) {
        const trainingData = [];

        demonstrations.forEach((demo) => {
            demo.steps.forEach((step) => {
                if (step.gameState && step.action) {
                    const input = this.stateToFeatures(step.gameState);
                    const output = this.actionToOutput(step.action);

                    trainingData.push({
                        input: input,
                        output: output,
                    });

                    // Data augmentation with noise
                    if (this.noiseFactor > 0) {
                        const noisyInput = input.map(
                            (val) =>
                                val + (Math.random() - 0.5) * this.noiseFactor,
                        );
                        trainingData.push({
                            input: noisyInput,
                            output: output,
                        });
                    }
                }
            });
        });

        return trainingData;
    }

    /**
     * Split data into training and validation sets
     */
    splitData(data) {
        const shuffled = [...data].sort(() => Math.random() - 0.5);
        const splitIndex = Math.floor(data.length * (1 - this.validationSplit));

        return {
            training: shuffled.slice(0, splitIndex),
            validation: shuffled.slice(splitIndex),
        };
    }

    /**
     * Train the model on demonstrations
     */
    async train(demonstrations) {
        if (!demonstrations || demonstrations.length === 0) {
            console.error("No demonstrations provided for training");
            return false;
        }

        this.updateParameters();
        this.isTraining = true;
        this.currentEpoch = 0;
        this.trainingHistory = [];

        console.log(
            `Starting imitation learning training with ${demonstrations.length} demonstrations`,
        );

        // Prepare training data
        const allData = this.prepareTrainingData(demonstrations);
        if (allData.length === 0) {
            console.error("No valid training data found in demonstrations");
            this.isTraining = false;
            return false;
        }

        const { training, validation } = this.splitData(allData);
        console.log(
            `Training samples: ${training.length}, Validation samples: ${validation.length}`,
        );

        try {
            // Train the model
            const result = await this.trainModel(training, validation);

            this.isTraining = false;
            console.log("Imitation learning training completed");

            return result;
        } catch (error) {
            console.error("Error during imitation learning training:", error);
            this.isTraining = false;
            return false;
        }
    }

    /**
     * Train the neural network model
     */
    async trainModel(trainingData, validationData) {
        return new Promise((resolve) => {
            try {
                // Configure training options
                const trainingOptions = {
                    iterations: this.maxEpochs,
                    errorThresh: 0.005,
                    log: true,
                    logPeriod: 10,
                    learningRate: this.learningRate,
                    callback: (stats) => {
                        this.currentEpoch = stats.iterations;

                        // Calculate validation error if we have validation data
                        let validationError = 0;
                        if (validationData.length > 0) {
                            validationError =
                                this.calculateValidationError(validationData);
                        }

                        this.trainingHistory.push({
                            epoch: stats.iterations,
                            trainingError: stats.error,
                            validationError: validationError,
                        });

                        // Update UI
                        this.updateTrainingProgress(
                            stats.iterations,
                            stats.error,
                            validationError,
                        );

                        // Early stopping if validation error starts increasing
                        if (this.shouldEarlyStop()) {
                            console.log("Early stopping triggered");
                            return true; // Stop training
                        }

                        return false; // Continue training
                    },
                };

                // Start training
                this.model.train(trainingData, trainingOptions);

                console.log("Model training completed");
                resolve(true);
            } catch (error) {
                console.error("Error in model training:", error);
                resolve(false);
            }
        });
    }

    /**
     * Calculate validation error
     */
    calculateValidationError(validationData) {
        if (!validationData || validationData.length === 0) return 0;

        let totalError = 0;
        let count = 0;

        validationData.forEach((sample) => {
            try {
                const prediction = this.model.run(sample.input);
                const error = this.calculateMSE(prediction, sample.output);
                totalError += error;
                count++;
            } catch (e) {
                // Skip invalid samples
            }
        });

        return count > 0 ? totalError / count : 0;
    }

    /**
     * Calculate Mean Squared Error
     */
    calculateMSE(prediction, target) {
        if (!prediction || !target || prediction.length !== target.length)
            return 1;

        let sum = 0;
        for (let i = 0; i < prediction.length; i++) {
            const diff = prediction[i] - target[i];
            sum += diff * diff;
        }

        return sum / prediction.length;
    }

    /**
     * Check if early stopping should be applied
     */
    shouldEarlyStop() {
        if (this.trainingHistory.length < 20) return false;

        const recent = this.trainingHistory.slice(-10);
        const older = this.trainingHistory.slice(-20, -10);

        const recentAvg =
            recent.reduce((sum, h) => sum + h.validationError, 0) /
            recent.length;
        const olderAvg =
            older.reduce((sum, h) => sum + h.validationError, 0) / older.length;

        // Stop if validation error is increasing
        return recentAvg > olderAvg * 1.05;
    }

    /**
     * Update training progress in UI
     */
    updateTrainingProgress(epoch, trainingError, validationError) {
        // Update status message
        document.getElementById("statusMessage").textContent =
            `Entraînement Imitation - Époque ${epoch}/${this.maxEpochs}`;

        // Update progress bar
        const progress = (epoch / this.maxEpochs) * 100;
        document.getElementById("waveProgressBar").style.width = `${progress}%`;
        document.getElementById("waveProgressText").textContent =
            `${Math.round(progress)}%`;

        console.log(
            `Epoch ${epoch}: Training Error = ${trainingError.toFixed(4)}, Validation Error = ${validationError.toFixed(4)}`,
        );
    }

    /**
     * Predict action for given game state
     */
    predict(gameState) {
        if (!this.model) {
            console.error("Model not initialized");
            return null;
        }

        try {
            const features = this.stateToFeatures(gameState);
            const output = this.model.run(features);
            return this.outputToAction(output);
        } catch (error) {
            console.error("Error in prediction:", error);
            return null;
        }
    }

    /**
     * Evaluate model performance on test data
     */
    evaluate(testDemonstrations) {
        if (
            !this.model ||
            !testDemonstrations ||
            testDemonstrations.length === 0
        ) {
            return { accuracy: 0, loss: 1 };
        }

        const testData = this.prepareTrainingData(testDemonstrations);
        let correct = 0;
        let totalLoss = 0;
        let total = 0;

        testData.forEach((sample) => {
            try {
                const prediction = this.model.run(sample.input);
                const predictedAction = this.outputToAction(prediction);
                const actualAction = this.outputToAction(sample.output);

                if (predictedAction === actualAction) {
                    correct++;
                }

                totalLoss += this.calculateMSE(prediction, sample.output);
                total++;
            } catch (e) {
                // Skip invalid samples
            }
        });

        return {
            accuracy: total > 0 ? correct / total : 0,
            loss: total > 0 ? totalLoss / total : 1,
        };
    }

    /**
     * Save model to JSON
     */
    saveModel() {
        if (!this.model) {
            console.error("No model to save");
            return null;
        }

        try {
            return {
                modelData: this.model.toJSON(),
                parameters: {
                    hiddenLayers: this.hiddenLayers,
                    learningRate: this.learningRate,
                    batchSize: this.batchSize,
                    maxEpochs: this.maxEpochs,
                    noiseFactor: this.noiseFactor,
                },
                trainingHistory: this.trainingHistory,
            };
        } catch (error) {
            console.error("Error saving model:", error);
            return null;
        }
    }

    /**
     * Load model from JSON
     */
    loadModel(modelData) {
        try {
            this.model = new brain.NeuralNetwork();
            this.model.fromJSON(modelData.modelData);

            if (modelData.parameters) {
                this.hiddenLayers =
                    modelData.parameters.hiddenLayers || this.hiddenLayers;
                this.learningRate =
                    modelData.parameters.learningRate || this.learningRate;
                this.batchSize =
                    modelData.parameters.batchSize || this.batchSize;
                this.maxEpochs =
                    modelData.parameters.maxEpochs || this.maxEpochs;
                this.noiseFactor =
                    modelData.parameters.noiseFactor || this.noiseFactor;
            }

            if (modelData.trainingHistory) {
                this.trainingHistory = modelData.trainingHistory;
            }

            console.log("Imitation learning model loaded successfully");
            return true;
        } catch (error) {
            console.error("Error loading model:", error);
            return false;
        }
    }

    /**
     * Get training statistics
     */
    getTrainingStats() {
        if (this.trainingHistory.length === 0) {
            return null;
        }

        const lastHistory =
            this.trainingHistory[this.trainingHistory.length - 1];

        return {
            epochs: this.trainingHistory.length,
            finalTrainingError: lastHistory.trainingError,
            finalValidationError: lastHistory.validationError,
            bestValidationError: Math.min(
                ...this.trainingHistory.map((h) => h.validationError),
            ),
            convergenceEpoch: this.findConvergenceEpoch(),
        };
    }

    /**
     * Find the epoch where the model converged
     */
    findConvergenceEpoch() {
        if (this.trainingHistory.length < 10)
            return this.trainingHistory.length;

        for (let i = 10; i < this.trainingHistory.length; i++) {
            const recent = this.trainingHistory.slice(i - 10, i);
            const avgError =
                recent.reduce((sum, h) => sum + h.trainingError, 0) /
                recent.length;

            if (this.trainingHistory[i].trainingError > avgError * 0.95) {
                return i;
            }
        }

        return this.trainingHistory.length;
    }

    /**
     * Reset the model
     */
    reset() {
        this.initializeModel();
        this.trainingHistory = [];
        this.currentEpoch = 0;
        this.isTraining = false;
    }
}
