/**
 * Demonstration Recorder for Imitation Learning
 * Records expert demonstrations and manages demonstration data
 */
class DemonstrationRecorder {
    constructor() {
        this.demonstrations = [];
        this.currentDemo = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.gameState = null;
        this.lastAction = null;
        this.actionBuffer = [];
        this.keyStates = {};
        this.recordingQuality = 'high'; // high, medium, low
        this.maxDemoLength = 1000; // Maximum steps per demonstration
        this.minActionInterval = 50; // Minimum milliseconds between recorded actions
        this.lastRecordTime = 0;

        this.initializeKeyListener();
    }

    /**
     * Initialize keyboard event listeners
     */
    initializeKeyListener() {
        // Track key states to avoid duplicate events
        this.validKeys = new Set([
            'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
            'w', 'a', 's', 'd', 'z', 'q' // WASD and AZERTY support
        ]);
    }

    /**
     * Start recording a new demonstration
     */
    startRecording() {
        if (this.isRecording) {
            console.warn('Already recording a demonstration');
            return false;
        }

        this.isRecording = true;
        this.recordingStartTime = Date.now();
        this.currentDemo = {
            id: this.generateDemoId(),
            timestamp: this.recordingStartTime,
            steps: [],
            metadata: {
                quality: this.recordingQuality,
                duration: 0,
                totalActions: 0,
                gameSettings: this.getCurrentGameSettings()
            }
        };

        console.log('Started recording demonstration:', this.currentDemo.id);
        return true;
    }

    /**
     * Stop recording current demonstration
     */
    stopRecording() {
        if (!this.isRecording || !this.currentDemo) {
            console.warn('No active recording to stop');
            return null;
        }

        this.isRecording = false;
        this.currentDemo.metadata.duration = Date.now() - this.recordingStartTime;
        this.currentDemo.metadata.totalActions = this.currentDemo.steps.length;

        // Validate demonstration quality
        if (this.validateDemonstration(this.currentDemo)) {
            this.demonstrations.push(this.currentDemo);
            console.log(`Demonstration recorded: ${this.currentDemo.steps.length} steps in ${this.currentDemo.metadata.duration}ms`);
        } else {
            console.warn('Demonstration failed validation and was discarded');
        }

        const completedDemo = this.currentDemo;
        this.currentDemo = null;
        this.actionBuffer = [];
        this.keyStates = {};

        return completedDemo;
    }

    /**
     * Record an action with current game state
     */
    recordAction(key, eventType, gameState = null) {
        if (!this.isRecording || !this.currentDemo) {
            return false;
        }

        // Check if this is a valid key
        if (!this.validKeys.has(key)) {
            return false;
        }

        const currentTime = Date.now();

        // Rate limiting to avoid too frequent recordings
        if (currentTime - this.lastRecordTime < this.minActionInterval) {
            return false;
        }

        // Update key states
        if (eventType === 'keydown') {
            if (this.keyStates[key]) {
                return false; // Key already pressed, avoid duplicate
            }
            this.keyStates[key] = true;
        } else if (eventType === 'keyup') {
            this.keyStates[key] = false;
            return false; // We only record keydown events for actions
        }

        // Get current game state
        const currentGameState = gameState || this.getCurrentGameState();
        if (!currentGameState) {
            return false;
        }

        // Create step record
        const step = {
            timestamp: currentTime - this.recordingStartTime,
            action: this.normalizeAction(key),
            gameState: this.cloneGameState(currentGameState),
            keyStates: { ...this.keyStates },
            metadata: {
                rawKey: key,
                eventType: eventType,
                reactionTime: this.calculateReactionTime(currentGameState)
            }
        };

        this.currentDemo.steps.push(step);
        this.lastRecordTime = currentTime;
        this.lastAction = step.action;

        // Prevent demonstration from becoming too long
        if (this.currentDemo.steps.length >= this.maxDemoLength) {
            console.warn('Maximum demonstration length reached, stopping recording');
            this.stopRecording();
        }

        return true;
    }

    /**
     * Normalize action names for consistency
     */
    normalizeAction(key) {
        const actionMap = {
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right',
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right',
            'z': 'up', // French AZERTY
            'q': 'left' // French AZERTY
        };

        return actionMap[key] || key.toLowerCase();
    }

    /**
     * Get current game state (this should be provided by the main game)
     */
    getCurrentGameState() {
        // This would be integrated with the main game system
        // For now, return a placeholder that matches expected structure
        return this.gameState || {
            agentX: 375,
            agentY: 375,
            velocityX: 0,
            velocityY: 0,
            circles: [],
            gameWidth: 750,
            gameHeight: 750,
            score: 0,
            timeRemaining: 5000,
            maxTime: 5000,
            initialCircles: 5
        };
    }

    /**
     * Update the current game state (called by main game loop)
     */
    updateGameState(gameState) {
        this.gameState = gameState;
    }

    /**
     * Clone game state to avoid reference issues
     */
    cloneGameState(gameState) {
        return {
            agentX: gameState.agentX,
            agentY: gameState.agentY,
            velocityX: gameState.velocityX || 0,
            velocityY: gameState.velocityY || 0,
            circles: gameState.circles ? gameState.circles.map(circle => ({
                x: circle.x,
                y: circle.y,
                radius: circle.radius,
                id: circle.id
            })) : [],
            gameWidth: gameState.gameWidth,
            gameHeight: gameState.gameHeight,
            score: gameState.score || 0,
            timeRemaining: gameState.timeRemaining || 0,
            maxTime: gameState.maxTime || 5000,
            initialCircles: gameState.initialCircles || 5
        };
    }

    /**
     * Calculate reaction time based on game state changes
     */
    calculateReactionTime(gameState) {
        if (!this.currentDemo || this.currentDemo.steps.length === 0) {
            return 0;
        }

        const lastStep = this.currentDemo.steps[this.currentDemo.steps.length - 1];
        const timeDiff = Date.now() - this.recordingStartTime - lastStep.timestamp;

        // Simple heuristic: reaction time based on state change
        if (this.hasSignificantStateChange(lastStep.gameState, gameState)) {
            return timeDiff;
        }

        return 0;
    }

    /**
     * Check if there's a significant change in game state
     */
    hasSignificantStateChange(oldState, newState) {
        if (!oldState || !newState) return false;

        // Check for position changes
        const positionChange = Math.abs(oldState.agentX - newState.agentX) > 5 ||
                              Math.abs(oldState.agentY - newState.agentY) > 5;

        // Check for circle count changes (collection)
        const circleCountChange = (oldState.circles?.length || 0) !== (newState.circles?.length || 0);

        return positionChange || circleCountChange;
    }

    /**
     * Validate demonstration quality
     */
    validateDemonstration(demo) {
        if (!demo || !demo.steps || demo.steps.length === 0) {
            return false;
        }

        // Minimum length requirement
        if (demo.steps.length < 10) {
            console.warn('Demonstration too short:', demo.steps.length, 'steps');
            return false;
        }

        // Check for reasonable action variety
        const actions = demo.steps.map(step => step.action);
        const uniqueActions = new Set(actions);
        if (uniqueActions.size < 2) {
            console.warn('Demonstration lacks action variety');
            return false;
        }

        // Check for realistic timing
        const totalTime = demo.metadata.duration;
        const avgActionInterval = totalTime / demo.steps.length;
        if (avgActionInterval < 10 || avgActionInterval > 2000) {
            console.warn('Demonstration has unrealistic timing:', avgActionInterval, 'ms per action');
            return false;
        }

        return true;
    }

    /**
     * Generate unique demonstration ID
     */
    generateDemoId() {
        return `demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get current game settings for metadata
     */
    getCurrentGameSettings() {
        return {
            gameSize: document.getElementById('gameSizeInput')?.value || 750,
            squareSize: document.getElementById('squareSizeInput')?.value || 25,
            circleRadius: document.getElementById('circleRadiusInput')?.value || 10,
            initialCircles: document.getElementById('initialCirclesInput')?.value || 5,
            waveDuration: document.getElementById('waveDurationInput')?.value || 5000
        };
    }

    /**
     * Load demonstrations from JSON data
     */
    loadDemonstrations(data) {
        try {
            if (Array.isArray(data)) {
                // Validate each demonstration
                const validDemos = data.filter(demo => this.validateDemonstration(demo));
                this.demonstrations = validDemos;
                console.log(`Loaded ${validDemos.length} valid demonstrations`);
            } else if (data.demonstrations && Array.isArray(data.demonstrations)) {
                const validDemos = data.demonstrations.filter(demo => this.validateDemonstration(demo));
                this.demonstrations = validDemos;
                console.log(`Loaded ${validDemos.length} valid demonstrations`);
            } else {
                throw new Error('Invalid demonstration data format');
            }

            return true;
        } catch (error) {
            console.error('Error loading demonstrations:', error);
            return false;
        }
    }

    /**
     * Export demonstrations to JSON
     */
    exportDemonstrations() {
        return {
            demonstrations: this.demonstrations,
            metadata: {
                exportTime: Date.now(),
                totalDemonstrations: this.demonstrations.length,
                recordingQuality: this.recordingQuality,
                version: '1.0'
            }
        };
    }

    /**
     * Clear all demonstrations
     */
    clearDemonstrations() {
        this.demonstrations = [];
        this.currentDemo = null;
        this.isRecording = false;
        console.log('All demonstrations cleared');
    }

    /**
     * Remove specific demonstration
     */
    removeDemonstration(demoId) {
        const index = this.demonstrations.findIndex(demo => demo.id === demoId);
        if (index >= 0) {
            this.demonstrations.splice(index, 1);
            console.log('Demonstration removed:', demoId);
            return true;
        }
        return false;
    }

    /**
     * Get demonstration statistics
     */
    getDemonstrationStats() {
        if (this.demonstrations.length === 0) {
            return null;
        }

        const totalSteps = this.demonstrations.reduce((sum, demo) => sum + demo.steps.length, 0);
        const totalDuration = this.demonstrations.reduce((sum, demo) => sum + demo.metadata.duration, 0);
        const avgStepsPerDemo = totalSteps / this.demonstrations.length;
        const avgDuration = totalDuration / this.demonstrations.length;

        // Action distribution
        const allActions = [];
        this.demonstrations.forEach(demo => {
            demo.steps.forEach(step => {
                allActions.push(step.action);
            });
        });

        const actionCounts = {};
        allActions.forEach(action => {
            actionCounts[action] = (actionCounts[action] || 0) + 1;
        });

        return {
            totalDemonstrations: this.demonstrations.length,
            totalSteps: totalSteps,
            totalDuration: totalDuration,
            avgStepsPerDemo: Math.round(avgStepsPerDemo),
            avgDuration: Math.round(avgDuration),
            actionDistribution: actionCounts,
            qualityScores: this.calculateQualityScores()
        };
    }

    /**
     * Calculate quality scores for demonstrations
     */
    calculateQualityScores() {
        return this.demonstrations.map(demo => {
            let score = 100;

            // Penalize very short or very long demonstrations
            if (demo.steps.length < 20) score -= 20;
            if (demo.steps.length > 500) score -= 10;

            // Reward action variety
            const actions = demo.steps.map(step => step.action);
            const uniqueActions = new Set(actions);
            score += (uniqueActions.size - 1) * 5;

            // Penalize rapid repeated actions (likely spam)
            let rapidRepeats = 0;
            for (let i = 1; i < demo.steps.length; i++) {
                if (demo.steps[i].action === demo.steps[i-1].action &&
                    demo.steps[i].timestamp - demo.steps[i-1].timestamp < 100) {
                    rapidRepeats++;
                }
            }
            score -= rapidRepeats * 2;

            return {
                demoId: demo.id,
                score: Math.max(0, Math.min(100, score)),
                steps: demo.steps.length,
                duration: demo.metadata.duration
            };
        });
    }

    /**
     * Get high-quality demonstrations only
     */
    getHighQualityDemonstrations(minScore = 70) {
        const qualityScores = this.calculateQualityScores();
        const highQualityIds = qualityScores
            .filter(score => score.score >= minScore)
            .map(score => score.demoId);

        return this.demonstrations.filter(demo => highQualityIds.includes(demo.id));
    }

    /**
     * Set recording quality
     */
    setRecordingQuality(quality) {
        const validQualities = ['high', 'medium', 'low'];
        if (validQualities.includes(quality)) {
            this.recordingQuality = quality;

            // Adjust parameters based on quality
            switch (quality) {
                case 'high':
                    this.minActionInterval = 50;
                    this.maxDemoLength = 1000;
                    break;
                case 'medium':
                    this.minActionInterval = 100;
                    this.maxDemoLength = 500;
                    break;
                case 'low':
                    this.minActionInterval = 200;
                    this.maxDemoLength = 250;
                    break;
            }

            console.log(`Recording quality set to: ${quality}`);
        }
    }
}
