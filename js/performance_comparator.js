/**
 * Performance Comparator for Multi-Modal Learning
 * Compares and visualizes performance across different learning methods
 */
class PerformanceComparator {
    constructor() {
        this.performanceData = {
            genetic: {
                scores: [],
                avgScore: 0,
                bestScore: 0,
                generations: 0,
                convergenceGeneration: -1,
                trainingTime: 0,
                parameters: {}
            },
            qlearning: {
                scores: [],
                avgScore: 0,
                bestScore: 0,
                episodes: 0,
                convergenceEpisode: -1,
                trainingTime: 0,
                parameters: {},
                epsilon: 1.0
            },
            imitation: {
                scores: [],
                avgScore: 0,
                bestScore: 0,
                epochs: 0,
                convergenceEpoch: -1,
                trainingTime: 0,
                parameters: {},
                trainingLoss: [],
                validationLoss: []
            },
            actorcritic: {
                scores: [],
                avgScore: 0,
                bestScore: 0,
                episodes: 0,
                convergenceEpisode: -1,
                trainingTime: 0,
                parameters: {},
                epsilon: 1.0,
                actorLoss: [],
                criticLoss: []
            }
        };

        this.chartCanvas = document.getElementById('performanceChart');
        this.chartContext = this.chartCanvas ? this.chartCanvas.getContext('2d') : null;
        this.updateInterval = null;
        this.isRealTimeUpdate = false;

        this.initializeChart();
        this.startRealTimeUpdates();
    }

    /**
     * Initialize the performance chart
     */
    initializeChart() {
        if (!this.chartContext) return;

        this.chartContext.fillStyle = '#1a202c';
        this.chartContext.fillRect(0, 0, this.chartCanvas.width, this.chartCanvas.height);

        // Draw initial empty chart
        this.drawChart();
    }

    /**
     * Start real-time performance updates
     */
    startRealTimeUpdates() {
        if (this.updateInterval) return;

        this.isRealTimeUpdate = true;
        this.updateInterval = setInterval(() => {
            this.updateDisplay();
        }, 1000); // Update every second
    }

    /**
     * Stop real-time updates
     */
    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        this.isRealTimeUpdate = false;
    }

    /**
     * Record performance data for a specific learning method
     */
    recordPerformance(method, data) {
        if (!this.performanceData[method]) {
            console.error('Unknown learning method:', method);
            return false;
        }

        const methodData = this.performanceData[method];

        // Update scores
        if (data.score !== undefined) {
            methodData.scores.push(data.score);
            methodData.bestScore = Math.max(methodData.bestScore, data.score);
            methodData.avgScore = this.calculateAverage(methodData.scores);
        }

        // Update method-specific data
        switch (method) {
            case 'genetic':
                if (data.generation !== undefined) methodData.generations = data.generation;
                if (data.convergenceGeneration !== undefined) methodData.convergenceGeneration = data.convergenceGeneration;
                break;

            case 'qlearning':
                if (data.episode !== undefined) methodData.episodes = data.episode;
                if (data.epsilon !== undefined) methodData.epsilon = data.epsilon;
                if (data.convergenceEpisode !== undefined) methodData.convergenceEpisode = data.convergenceEpisode;
                break;

            case 'imitation':
                if (data.epoch !== undefined) methodData.epochs = data.epoch;
                if (data.trainingLoss !== undefined) methodData.trainingLoss.push(data.trainingLoss);
                if (data.validationLoss !== undefined) methodData.validationLoss.push(data.validationLoss);
                if (data.convergenceEpoch !== undefined) methodData.convergenceEpoch = data.convergenceEpoch;
                break;

            case 'actorcritic':
                if (data.episode !== undefined) methodData.episodes = data.episode;
                if (data.epsilon !== undefined) methodData.epsilon = data.epsilon;
                if (data.convergenceEpisode !== undefined) methodData.convergenceEpisode = data.convergenceEpisode;
                if (data.actorLoss !== undefined) methodData.actorLoss.push(data.actorLoss);
                if (data.criticLoss !== undefined) methodData.criticLoss.push(data.criticLoss);
                break;
        }

        // Update training time
        if (data.trainingTime !== undefined) {
            methodData.trainingTime = data.trainingTime;
        }

        // Update parameters
        if (data.parameters) {
            methodData.parameters = { ...methodData.parameters, ...data.parameters };
        }

        return true;
    }

    /**
     * Calculate average of an array
     */
    calculateAverage(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((sum, val) => sum + val, 0) / arr.length;
    }

    /**
     * Update the performance display
     */
    updateDisplay() {
        // Update score displays
        document.getElementById('geneticScore').textContent =
            this.performanceData.genetic.bestScore.toFixed(0);
        document.getElementById('qlearningScore').textContent =
            this.performanceData.qlearning.bestScore.toFixed(0);
        document.getElementById('imitationScore').textContent =
            this.performanceData.imitation.bestScore.toFixed(0);
        document.getElementById('actorCriticScore').textContent =
            this.performanceData.actorcritic.bestScore.toFixed(0);

        // Update chart
        this.drawChart();
    }

    /**
     * Draw the performance comparison chart
     */
    drawChart() {
        if (!this.chartContext) return;

        const canvas = this.chartCanvas;
        const ctx = this.chartContext;

        // Clear canvas
        ctx.fillStyle = '#1a202c';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Get data for visualization
        const maxDataPoints = 50;
        const geneticScores = this.getRecentScores('genetic', maxDataPoints);
        const qlearningScores = this.getRecentScores('qlearning', maxDataPoints);
        const imitationScores = this.getRecentScores('imitation', maxDataPoints);
        const actorCriticScores = this.getRecentScores('actorcritic', maxDataPoints);

        // Find global max for scaling
        const allScores = [...geneticScores, ...qlearningScores, ...imitationScores, ...actorCriticScores];
        const maxScore = Math.max(...allScores, 1);
        const minScore = Math.min(...allScores, 0);
        const scoreRange = maxScore - minScore || 1;

        // Chart dimensions
        const padding = 40;
        const chartWidth = canvas.width - 2 * padding;
        const chartHeight = canvas.height - 2 * padding;

        // Draw axes
        this.drawAxes(ctx, padding, chartWidth, chartHeight, minScore, maxScore);

        // Draw performance lines
        this.drawPerformanceLine(ctx, geneticScores, '#3b82f6', padding, chartWidth, chartHeight, minScore, scoreRange);
        this.drawPerformanceLine(ctx, qlearningScores, '#10b981', padding, chartWidth, chartHeight, minScore, scoreRange);
        this.drawPerformanceLine(ctx, imitationScores, '#8b5cf6', padding, chartWidth, chartHeight, minScore, scoreRange);
        this.drawPerformanceLine(ctx, actorCriticScores, '#ef4444', padding, chartWidth, chartHeight, minScore, scoreRange);

        // Draw legend
        this.drawLegend(ctx, canvas.width, canvas.height);
    }

    /**
     * Get recent scores for a method
     */
    getRecentScores(method, maxPoints) {
        const scores = this.performanceData[method].scores;
        if (scores.length <= maxPoints) {
            return scores;
        }
        return scores.slice(-maxPoints);
    }

    /**
     * Draw chart axes
     */
    drawAxes(ctx, padding, chartWidth, chartHeight, minScore, maxScore) {
        ctx.strokeStyle = '#4a5568';
        ctx.lineWidth = 1;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, padding + chartHeight);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding, padding + chartHeight);
        ctx.lineTo(padding + chartWidth, padding + chartHeight);
        ctx.stroke();

        // Y-axis labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px Arial';
        ctx.textAlign = 'right';

        for (let i = 0; i <= 5; i++) {
            const y = padding + chartHeight - (i / 5) * chartHeight;
            const value = minScore + (i / 5) * (maxScore - minScore);
            ctx.fillText(value.toFixed(0), padding - 5, y + 3);
        }

        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Progression', padding + chartWidth / 2, padding + chartHeight + 30);

        // Y-axis label
        ctx.save();
        ctx.translate(15, padding + chartHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Score', 0, 0);
        ctx.restore();
    }

    /**
     * Draw performance line for a method
     */
    drawPerformanceLine(ctx, scores, color, padding, chartWidth, chartHeight, minScore, scoreRange) {
        if (scores.length < 2) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        scores.forEach((score, index) => {
            const x = padding + (index / (scores.length - 1)) * chartWidth;
            const y = padding + chartHeight - ((score - minScore) / scoreRange) * chartHeight;

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw points
        ctx.fillStyle = color;
        scores.forEach((score, index) => {
            if (index % Math.max(1, Math.floor(scores.length / 20)) === 0) {
                const x = padding + (index / (scores.length - 1)) * chartWidth;
                const y = padding + chartHeight - ((score - minScore) / scoreRange) * chartHeight;

                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }

    /**
     * Draw chart legend
     */
    drawLegend(ctx, canvasWidth, canvasHeight) {
        const legendItems = [
            { label: 'Génétique', color: '#3b82f6' },
            { label: 'Q-Learning', color: '#10b981' },
            { label: 'Imitation', color: '#8b5cf6' },
            { label: 'Actor-Critic', color: '#ef4444' }
        ];

        ctx.font = '12px Arial';
        ctx.textAlign = 'left';

        legendItems.forEach((item, index) => {
            const x = canvasWidth - 100;
            const y = 20 + index * 20;

            // Draw color box
            ctx.fillStyle = item.color;
            ctx.fillRect(x, y - 8, 12, 12);

            // Draw label
            ctx.fillStyle = '#e5e7eb';
            ctx.fillText(item.label, x + 18, y + 2);
        });
    }

    /**
     * Compare methods and return winner
     */
    comparePerformance() {
        const methods = ['genetic', 'qlearning', 'imitation'];
        const results = methods.map(method => ({
            method: method,
            avgScore: this.performanceData[method].avgScore,
            bestScore: this.performanceData[method].bestScore,
            convergenceSpeed: this.getConvergenceSpeed(method),
            trainingTime: this.performanceData[method].trainingTime,
            efficiency: this.calculateEfficiency(method)
        }));

        // Sort by efficiency (best score / training time)
        results.sort((a, b) => b.efficiency - a.efficiency);

        return {
            winner: results[0],
            comparison: results,
            analysis: this.generateAnalysis(results)
        };
    }

    /**
     * Get convergence speed for a method
     */
    getConvergenceSpeed(method) {
        const data = this.performanceData[method];

        switch (method) {
            case 'genetic':
                return data.convergenceGeneration > 0 ? data.convergenceGeneration : data.generations;
            case 'qlearning':
                return data.convergenceEpisode > 0 ? data.convergenceEpisode : data.episodes;
            case 'imitation':
                return data.convergenceEpoch > 0 ? data.convergenceEpoch : data.epochs;
            default:
                return 0;
        }
    }

    /**
     * Calculate efficiency score
     */
    calculateEfficiency(method) {
        const data = this.performanceData[method];
        if (data.trainingTime === 0) return 0;

        return data.bestScore / (data.trainingTime / 1000); // Score per second
    }

    /**
     * Generate performance analysis
     */
    generateAnalysis(results) {
        const analysis = {
            bestOverall: results[0].method,
            fastestConvergence: '',
            mostConsistent: '',
            recommendations: []
        };

        // Find fastest convergence
        let minConvergence = Infinity;
        results.forEach(result => {
            if (result.convergenceSpeed < minConvergence && result.convergenceSpeed > 0) {
                minConvergence = result.convergenceSpeed;
                analysis.fastestConvergence = result.method;
            }
        });

        // Find most consistent (lowest variance in recent scores)
        let minVariance = Infinity;
        ['genetic', 'qlearning', 'imitation', 'actorcritic'].forEach(method => {
            const recentScores = this.getRecentScores(method, 10);
            if (recentScores.length > 5) {
                const variance = this.calculateVariance(recentScores);
                if (variance < minVariance) {
                    minVariance = variance;
                    analysis.mostConsistent = method;
                }
            }
        });

        // Generate recommendations
        if (results[0].bestScore > results[1].bestScore * 1.2) {
            analysis.recommendations.push(`${results[0].method} shows significantly better performance`);
        }

        if (analysis.fastestConvergence !== analysis.bestOverall) {
            analysis.recommendations.push(`Consider ${analysis.fastestConvergence} for faster training`);
        }

        return analysis;
    }

    /**
     * Calculate variance of scores
     */
    calculateVariance(scores) {
        if (scores.length < 2) return 0;

        const mean = this.calculateAverage(scores);
        const squaredDiffs = scores.map(score => Math.pow(score - mean, 2));
        return this.calculateAverage(squaredDiffs);
    }

    /**
     * Export performance data
     */
    exportPerformanceData() {
        const exportData = {
            timestamp: Date.now(),
            performanceData: this.performanceData,
            comparison: this.comparePerformance(),
            metadata: {
                version: '1.0',
                totalTrainingTime: Object.values(this.performanceData)
                    .reduce((sum, data) => sum + data.trainingTime, 0)
            }
        };

        return JSON.stringify(exportData, null, 2);
    }

    /**
     * Import performance data
     */
    importPerformanceData(jsonData) {
        try {
            const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;

            if (data.performanceData) {
                this.performanceData = { ...this.performanceData, ...data.performanceData };
                this.updateDisplay();
                console.log('Performance data imported successfully');
                return true;
            }

            return false;
        } catch (error) {
            console.error('Error importing performance data:', error);
            return false;
        }
    }

    /**
     * Reset all performance data
     */
    resetPerformanceData() {
        Object.keys(this.performanceData).forEach(method => {
            this.performanceData[method] = {
                scores: [],
                avgScore: 0,
                bestScore: 0,
                generations: 0,
                episodes: 0,
                epochs: 0,
                convergenceGeneration: -1,
                convergenceEpisode: -1,
                convergenceEpoch: -1,
                trainingTime: 0,
                parameters: {},
                ...(method === 'qlearning' && { epsilon: 1.0 }),
                ...(method === 'imitation' && { trainingLoss: [], validationLoss: [] })
            };
        });

        this.updateDisplay();
        console.log('Performance data reset');
    }

    /**
     * Get detailed statistics for a method
     */
    getMethodStatistics(method) {
        if (!this.performanceData[method]) return null;

        const data = this.performanceData[method];
        const recentScores = this.getRecentScores(method, 20);

        return {
            method: method,
            totalRuns: data.scores.length,
            bestScore: data.bestScore,
            avgScore: data.avgScore,
            recentAvgScore: this.calculateAverage(recentScores),
            variance: this.calculateVariance(data.scores),
            trainingTime: data.trainingTime,
            efficiency: this.calculateEfficiency(method),
            convergenceSpeed: this.getConvergenceSpeed(method),
            parameters: data.parameters
        };
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopRealTimeUpdates();
        this.resetPerformanceData();
    }
}
