document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const canvas = document.getElementById('gridCanvas');
    const startPauseButton = document.getElementById('startPauseButton');
    const resetButton = document.getElementById('resetButton');

    const episodeDisplay = document.getElementById('episodeDisplay');
    const stepsDisplay = document.getElementById('stepsDisplay');
    const rewardDisplay = document.getElementById('rewardDisplay');
    const epsilonDisplay = document.getElementById('epsilonDisplay');
    const statusDisplay = document.getElementById('statusDisplay');

    const gridSizeInput = document.getElementById('gridSize');
    const trainingSpeedInput = document.getElementById('trainingSpeed');
    const numPointsInput = document.getElementById('numPoints');
    const maxStepsInput = document.getElementById('maxSteps');
    const numEpisodesInput = document.getElementById('numEpisodes');
    const alphaInput = document.getElementById('alpha');
    const gammaInput = document.getElementById('gamma');
    const epsilonStartInput = document.getElementById('epsilonStart');
    const epsilonDecayInput = document.getElementById('epsilonDecay');

    // Manual Control Buttons
    const manualUpButton = document.getElementById('manualUp');
    const manualDownButton = document.getElementById('manualDown');
    const manualLeftButton = document.getElementById('manualLeft');
    const manualRightButton = document.getElementById('manualRight');
    const manualStayButton = document.getElementById('manualStay');

    // Q-Table Management Buttons
    const saveQTableButton = document.getElementById('saveQTableButton');
    const loadQTableButton = document.getElementById('loadQTableButton');
    const clearQTableButton = document.getElementById('clearQTableButton');

    // Heatmap Canvas (for future use)
    const heatmapCanvas = document.getElementById('heatmapCanvas');
    const heatmapCtx = heatmapCanvas.getContext('2d');

    // --- Game and Agent Initialization ---
    let gridSize = parseInt(gridSizeInput.value);
    let numPoints = parseInt(numPointsInput.value);

    // Ensure canvas is sized by CSS first, then get its dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    heatmapCanvas.width = heatmapCanvas.clientWidth;
    heatmapCanvas.height = heatmapCanvas.clientHeight;


    let environment = new Environment(gridSize, 'gridCanvas', numPoints);
    let agent = new QLearningAgent(
        environment.actions, // All possible actions
        parseFloat(alphaInput.value),
        parseFloat(gammaInput.value),
        parseFloat(epsilonStartInput.value),
        parseFloat(epsilonDecayInput.value)
    );

    // --- Training State ---
    let trainingIntervalId = null;
    let currentEpisode = 0;
    let currentStep = 0;
    let totalReward = 0;
    let isPaused = false;
    let maxEpisodes;
    let maxStepsPerEpisode;
    let trainingSpeed;

    // --- UI Update Functions ---
    function updateInfoPanel() {
        episodeDisplay.textContent = currentEpisode;
        stepsDisplay.textContent = currentStep;
        rewardDisplay.textContent = totalReward.toFixed(2);
        epsilonDisplay.textContent = agent.epsilon.toFixed(4);
    }

    function updateStatus(message) {
        statusDisplay.textContent = message;
    }

    // --- Core Training Logic ---
    function runTrainingStep() {
        if (isPaused || currentEpisode >= maxEpisodes) {
            stopTraining("Training finished or paused.");
            return;
        }

        if (currentStep >= maxStepsPerEpisode) {
            startNewEpisode();
            return;
        }

        const state = environment.getState();
        const action = agent.chooseAction(state);
        const { nextState, reward, done } = environment.step(action);

        agent.updateQValue(state, action, reward, nextState);

        totalReward += reward;
        currentStep++;
        updateInfoPanel();
        environment.draw(); // Re-draw environment after step

        if (done || environment.points.length === 0) { // Episode might end early if all points are collected
            // For now, we continue until maxSteps or all points are collected AND no new points are spawned.
            // If all points collected, we could optionally respawn them or end episode.
            // Current environment._placePoints() only happens at reset.
            // Let's consider ending episode if all points collected.
            if (environment.points.length === 0) {
                updateStatus(`All points collected in episode ${currentEpisode}!`);
                startNewEpisode();
            }
        }
    }

    function startNewEpisode() {
        currentEpisode++;
        currentStep = 0;
        // totalReward = 0; // Reset reward per episode or accumulate? Let's accumulate for now.
        environment.reset(); // Reset environment for new episode
        agent.decayEpsilon(); // Decay epsilon at the start of each new episode
        updateInfoPanel();
        if (currentEpisode >= maxEpisodes) {
            stopTraining("All episodes completed.");
        } else {
            updateStatus(`Episode ${currentEpisode} started.`);
        }
    }

    function startTraining() {
        if (trainingIntervalId !== null) { // Already running or paused
            if (isPaused) {
                isPaused = false;
                startPauseButton.textContent = "Pause";
                updateStatus("Training resumed...");
                trainingIntervalId = setInterval(runTrainingStep, trainingSpeed);
            }
            return;
        }

        maxEpisodes = parseInt(numEpisodesInput.value);
        maxStepsPerEpisode = parseInt(maxStepsInput.value);
        trainingSpeed = parseInt(trainingSpeedInput.value);

        // Update agent parameters from UI before starting
        agent.setParameters(
            parseFloat(alphaInput.value),
            parseFloat(gammaInput.value),
            parseFloat(epsilonStartInput.value),
            parseFloat(epsilonDecayInput.value)
        );
        agent.resetEpsilon(); // Reset epsilon to start value for a new training run

        currentEpisode = 0;
        totalReward = 0;
        isPaused = false;
        startNewEpisode(); // Initialize first episode

        startPauseButton.textContent = "Pause";
        updateStatus("Training started...");
        trainingIntervalId = setInterval(runTrainingStep, trainingSpeed);
    }

    function pauseTraining() {
        if (trainingIntervalId !== null && !isPaused) {
            isPaused = true;
            clearInterval(trainingIntervalId);
            // trainingIntervalId = null; // Keep it to distinguish from fully stopped
            startPauseButton.textContent = "Resume";
            updateStatus("Training paused.");
        }
    }

    function stopTraining(message) {
        if (trainingIntervalId !== null) {
            clearInterval(trainingIntervalId);
        }
        trainingIntervalId = null;
        isPaused = false;
        startPauseButton.textContent = "Start";
        updateStatus(message || "Training stopped/reset.");
        console.log("Q-Table size:", agent.qTable.size);
        drawHeatmap(); // For bonus
    }

    function resetTraining() {
        stopTraining("Training reset.");
        currentEpisode = 0;
        currentStep = 0;
        totalReward = 0;
        gridSize = parseInt(gridSizeInput.value);
        numPoints = parseInt(numPointsInput.value);
        environment.updateGridSize(gridSize);
        environment.updateNumPoints(numPoints);
        agent.clearQTable(); // Optionally clear Q-table on reset, or keep it for further training
        agent.resetEpsilon(); // Reset epsilon
        updateInfoPanel();
        environment.draw(); // Draw initial state
        drawHeatmap();
    }


    // --- Event Listeners ---
    startPauseButton.addEventListener('click', () => {
        if (trainingIntervalId === null || !isPaused && startPauseButton.textContent === "Start") {
            startTraining();
        } else if (isPaused) { // If paused, resume
             isPaused = false;
             startPauseButton.textContent = "Pause";
             updateStatus("Training resumed...");
             trainingIntervalId = setInterval(runTrainingStep, trainingSpeed); // Use stored speed
        } else { // If running, pause
            pauseTraining();
        }
    });

    resetButton.addEventListener('click', resetTraining);

    // Update environment/agent if parameters change
    gridSizeInput.addEventListener('change', () => {
        if (trainingIntervalId !== null) stopTraining("Parameters changed, stopping training.");
        gridSize = parseInt(gridSizeInput.value);
        environment.updateGridSize(gridSize);
        // Canvas may need resize here if aspect ratio changes or if cellSize logic is complex
        canvas.width = canvas.clientWidth; // re-set canvas internal size
        canvas.height = canvas.clientHeight;
        environment.cellSize = canvas.width / environment.gridSize; // Recalculate cell size
        environment.draw();
        // agent.clearQTable(); // Changing grid size invalidates Q-table states
        resetTraining(); // Full reset is safer
    });

    numPointsInput.addEventListener('change', () => {
        if (trainingIntervalId !== null) stopTraining("Parameters changed, stopping training.");
        numPoints = parseInt(numPointsInput.value);
        environment.updateNumPoints(numPoints);
        environment.draw();
        resetTraining(); // Full reset is safer as state representation might change
    });

    // For other parameters, they are read when training starts or can be applied dynamically
    alphaInput.addEventListener('change', () => agent.alpha = parseFloat(alphaInput.value));
    gammaInput.addEventListener('change', () => agent.gamma = parseFloat(gammaInput.value));
    epsilonStartInput.addEventListener('change', () => {
        agent.epsilonStart = parseFloat(epsilonStartInput.value);
        if (trainingIntervalId === null) agent.epsilon = agent.epsilonStart; // Update current if not training
        epsilonDisplay.textContent = agent.epsilon.toFixed(4);
    });
    epsilonDecayInput.addEventListener('change', () => agent.epsilonDecay = parseFloat(epsilonDecayInput.value));
    trainingSpeedInput.addEventListener('change', () => {
        trainingSpeed = parseInt(trainingSpeedInput.value);
        if (trainingIntervalId !== null && !isPaused) { // If training is active
            clearInterval(trainingIntervalId);
            trainingIntervalId = setInterval(runTrainingStep, trainingSpeed);
        }
    });


    // --- Manual Control ---
    function handleManualMove(action) {
        if (trainingIntervalId !== null) {
            updateStatus("Pause or reset training to use manual control.");
            return;
        }
        const result = environment.manualMove(action); // manualMove now calls draw()
        updateStatus(`Manual: Action ${action}, Reward ${result.reward.toFixed(1)}`);
        // No Q-table update for manual moves
    }

    manualUpButton.addEventListener('click', () => handleManualMove(environment.ACTION_SPACE.UP));
    manualDownButton.addEventListener('click', () => handleManualMove(environment.ACTION_SPACE.DOWN));
    manualLeftButton.addEventListener('click', () => handleManualMove(environment.ACTION_SPACE.LEFT));
    manualRightButton.addEventListener('click', () => handleManualMove(environment.ACTION_SPACE.RIGHT));
    manualStayButton.addEventListener('click', () => handleManualMove(environment.ACTION_SPACE.STAY));

    document.addEventListener('keydown', (e) => {
        // Prevent arrow keys from scrolling the page if canvas or body has focus
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
             // Check if focus is on an input field, if so, don't hijack keys
            if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
                return;
            }
            e.preventDefault();
        }

        if (trainingIntervalId !== null) return; // Only allow manual when not training

        switch (e.key) {
            case 'ArrowUp':
                handleManualMove(environment.ACTION_SPACE.UP);
                break;
            case 'ArrowDown':
                handleManualMove(environment.ACTION_SPACE.DOWN);
                break;
            case 'ArrowLeft':
                handleManualMove(environment.ACTION_SPACE.LEFT);
                break;
            case 'ArrowRight':
                handleManualMove(environment.ACTION_SPACE.RIGHT);
                break;
            // No key for 'STAY' typically, but can be added
        }
    });

    // --- Q-Table Management ---
    saveQTableButton.addEventListener('click', () => {
        const qTableData = agent.getQTable();
        try {
            localStorage.setItem('qTable', JSON.stringify(qTableData));
            updateStatus('Q-Table saved to localStorage.');
        } catch (e) {
            updateStatus('Error saving Q-Table. Storage might be full.');
            console.error("Error saving Q-Table:", e);
        }
    });

    loadQTableButton.addEventListener('click', () => {
        try {
            const storedQTable = localStorage.getItem('qTable');
            if (storedQTable) {
                const qTableData = JSON.parse(storedQTable);
                agent.loadQTable(qTableData);
                updateStatus('Q-Table loaded from localStorage.');
                drawHeatmap(); // Update heatmap if visible
            } else {
                updateStatus('No Q-Table found in localStorage.');
            }
        } catch (e) {
            updateStatus('Error loading Q-Table.');
            console.error("Error loading Q-Table:", e);
        }
    });

    clearQTableButton.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear the Q-Table? This cannot be undone.")) {
            agent.clearQTable();
            updateStatus('Q-Table cleared.');
            drawHeatmap();
        }
    });

    // --- Initial Setup ---
    updateInfoPanel();
    environment.draw(); // Draw initial state of the environment
    drawHeatmap(); // Draw initial heatmap (empty)

// --- Heatmap Functions (Add these NEW functions) ---
function getHeatmapColor(value, minQ, maxQ) {
    if (value === undefined || value === null || !isFinite(value)) return '#e0e0e0'; // Default for no/invalid Q-value

    if (minQ === maxQ) {
        if (value > 0) return 'hsl(120, 70%, 50%)'; // Green for positive
        if (value < 0) return 'hsl(0, 70%, 50%)';   // Red for negative
        return '#cccccc'; // Neutral grey
    }

    // Normalize value
    const percentage = (value - minQ) / (maxQ - minQ);

    if (value === 0) return '#f0f0f0'; // Light grey for zero values

    let hue, saturation, lightness;

    if (value > 0) { // Positive values: shades of green
        hue = 120; // Green
        saturation = 70;
        lightness = 90 - (percentage * 40); // Lighter for smaller positive, darker for larger positive
    } else { // Negative values: shades of red
        hue = 0; // Red
        saturation = 70;
        // For negative values, percentage will be < 0 if minQ is negative.
        // We want closer to zero to be lighter red, more negative to be darker red.
        // Let's adjust percentage for negative range: (value - minQ) / (0 - minQ) roughly
        const negPercentage = minQ < 0 ? (value - minQ) / (0 - minQ) : 0.5; // Avoid division by zero if minQ is 0
        lightness = 90 - ((1-negPercentage) * 40);
    }

    // Fallback for extreme cases or if logic above is tricky
    if (value > 0.01) return `rgba(0, 180, 0, ${Math.min(1, percentage * 1.5)})`;
    if (value < -0.01) return `rgba(180, 0, 0, ${Math.min(1, (1-percentage) * 1.5)})`;


    return `hsl(${hue}, ${saturation}%, ${Math.max(30, Math.min(90, lightness))}%)`;
}


function drawHeatmap() {
    if (!heatmapCanvas || !heatmapCtx || !agent || !environment) {
        console.warn("Heatmap drawing prerequisites not met.");
        return;
    }

    const hGridSize = environment.gridSize;
    if (hGridSize === 0) return; // Avoid division by zero if grid size is not set

    heatmapCanvas.width = heatmapCanvas.clientWidth; // Ensure internal resolution is updated
    heatmapCanvas.height = heatmapCanvas.clientHeight;
    const hCellSize = heatmapCanvas.width / hGridSize;

    heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

    let qValuesForScaling = [];
    for (let r = 0; r < hGridSize; r++) {
        for (let c = 0; c < hGridSize; c++) {
            const stateKey = `agent_${c}_${r}_no_points`;
            if (agent.qTable.has(stateKey)) {
                const qValues = agent.qTable.get(stateKey);
                const maxQ = Math.max(...qValues.filter(q => isFinite(q)));
                if (isFinite(maxQ)) {
                    qValuesForScaling.push(maxQ);
                }
            }
        }
    }

    const minActualQ = qValuesForScaling.length > 0 ? Math.min(...qValuesForScaling) : 0;
    const maxActualQ = qValuesForScaling.length > 0 ? Math.max(...qValuesForScaling) : 0;

    for (let r = 0; r < hGridSize; r++) { // row
        for (let c = 0; c < hGridSize; c++) { // col
            const stateKey = `agent_${c}_${r}_no_points`;
            let cellValue;
            if (agent.qTable.has(stateKey)) {
                const qValues = agent.qTable.get(stateKey);
                const filteredQValues = qValues.filter(q => isFinite(q));
                if (filteredQValues.length > 0) {
                     cellValue = Math.max(...filteredQValues);
                }
            }

            heatmapCtx.fillStyle = getHeatmapColor(cellValue, minActualQ, maxActualQ);
            heatmapCtx.fillRect(c * hCellSize, r * hCellSize, hCellSize, hCellSize);
        }
    }

    // Draw grid lines on heatmap
    heatmapCtx.beginPath();
    for (let i = 0; i <= hGridSize; i++) {
        heatmapCtx.moveTo(i * hCellSize, 0);
        heatmapCtx.lineTo(i * hCellSize, heatmapCanvas.height);
        heatmapCtx.moveTo(0, i * hCellSize);
        heatmapCtx.lineTo(heatmapCanvas.width, i * hCellSize);
    }
    heatmapCtx.strokeStyle = '#bbb'; // Slightly darker for better visibility
    heatmapCtx.stroke();

    const heatmapHeading = document.querySelector('.q-table-heatmap h2');
    if(heatmapHeading) {
        heatmapHeading.textContent = "Q-Table Heatmap (Max Q for state 'agent_x_y_no_points')";
    } else {
        console.warn("Heatmap heading element not found.");
    }
}

});
