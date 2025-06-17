// worker.js
// Ce script s'exécute dans un Web Worker.
// Il gère deux tâches : l'évaluation d'un seul agent et l'évolution de toute la population.

import {
	ActorCritic
} from './actorCritic.js';

// --- Fonctions de l'algorithme génétique (opèrent sur des Float32Array) ---

function crossover(parent1Weights, parent2Weights) {
	const childWeights = new Float32Array(parent1Weights);
	for (let i = 0; i < parent1Weights.length; i++) {
		if (Math.random() < 0.5) {
			childWeights[i] = parent2Weights[i];
		}
	}
	return childWeights;
}

function mutate(weights, mutationRate, mutationStrength = 0.1) {
	const mutatedWeights = new Float32Array(weights);
	for (let i = 0; i < mutatedWeights.length; i++) {
		if (Math.random() < mutationRate) {
			mutatedWeights[i] += (Math.random() * 2 - 1) * mutationStrength;
		}
	}
	return mutatedWeights;
}


// --- Logique d'évolution (tâche 'evolve') ---
function evolvePopulation(data) {
	const {
		populationData,
		gaParams
	} = data;
	const {
		numIndividuals,
		numElitism,
		mutationRate,
	} = gaParams;

	populationData.sort((a, b) => b.fitness - a.fitness);

	const nextPopulationBrains = [];

	// Élitisme
	for (let i = 0; i < Math.min(numElitism, numIndividuals); i++) {
		nextPopulationBrains.push(populationData[i].agentBrain);
	}

	// Sélection et Crossover
	const numParents = Math.ceil(numIndividuals * 0.5);
	const selectedParents = populationData.slice(0, numParents);

	while (nextPopulationBrains.length < numIndividuals) {
		const parent1Data = selectedParents[Math.floor(Math.random() * selectedParents.length)];
		const parent2Data = selectedParents[Math.floor(Math.random() * selectedParents.length)];

		const parent1ActorWeights = parent1Data.agentBrain.actorWeights;
		const parent1CriticWeights = parent1Data.agentBrain.criticWeights;
		const parent2ActorWeights = parent2Data.agentBrain.actorWeights;
		const parent2CriticWeights = parent2Data.agentBrain.criticWeights;

		let childActorWeights = crossover(parent1ActorWeights, parent2ActorWeights);
		let childCriticWeights = crossover(parent1CriticWeights, parent2CriticWeights);

		childActorWeights = mutate(childActorWeights, mutationRate);
		childCriticWeights = mutate(childCriticWeights, mutationRate);

		nextPopulationBrains.push({
			actorWeights: childActorWeights,
			criticWeights: childCriticWeights,
		});
	}

	return nextPopulationBrains;
}


// --- Logique de simulation (tâche 'evaluate') ---
// NOTE: Ces fonctions sont des copies de celles de votre projet original.
let circlesWorker, squareXWorker, squareYWorker;
let canvasWidthWorker, canvasHeightWorker, SQUARE_SIZE_WORKER, SQUARE_SPEED_WORKER, CIRCLE_RADIUS_WORKER, MAX_CIRCLES_WORKER;
let envConfigWorker = {};

// Variables d'état pour le calcul de récompense par épisode dans le worker
let lastSquareXEpisodeWorker, lastSquareYEpisodeWorker, inactivityStepsEpisodeWorker, lastActionIndexEpisodeWorker = -1, repetitiveActionStepsEpisodeWorker, episodeWallHitEpisodeWorker, actionHistoryBufferEpisodeWorker = [], previousDistanceToClosestCircleEpisodeWorker, stagnationTrackingStepsEpisodeWorker = 0;


function getRandomIntWorker(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generateCircleWorker() {
    let newCircleX, newCircleY, collision;
    let attempts = 0;
    const MAX_ATTEMPTS = 100;
    do {
        collision = false;
        newCircleX = getRandomIntWorker(CIRCLE_RADIUS_WORKER, canvasWidthWorker - CIRCLE_RADIUS_WORKER);
        newCircleY = getRandomIntWorker(CIRCLE_RADIUS_WORKER, canvasHeightWorker - CIRCLE_RADIUS_WORKER);
        for (const existingCircle of circlesWorker) {
            const dx = newCircleX - existingCircle.x;
            const dy = newCircleY - existingCircle.y;
            if (Math.sqrt(dx * dx + dy * dy) < CIRCLE_RADIUS_WORKER * 2) {
                collision = true;
                break;
            }
        }
        attempts++;
        if (attempts > MAX_ATTEMPTS) break;
    } while (collision);
    circlesWorker.push({ x: newCircleX, y: newCircleY });
}

function resetGameEnvironmentWorker() {
    squareXWorker = canvasWidthWorker / 2 - SQUARE_SIZE_WORKER / 2;
    squareYWorker = canvasHeightWorker / 2 - SQUARE_SIZE_WORKER / 2;
    circlesWorker = [];
    for (let i = 0; i < MAX_CIRCLES_WORKER; i++) {
        generateCircleWorker();
    }
    lastSquareXEpisodeWorker = squareXWorker;
    lastSquareYEpisodeWorker = squareYWorker;
    inactivityStepsEpisodeWorker = 0;
    lastActionIndexEpisodeWorker = -1;
    repetitiveActionStepsEpisodeWorker = 0;
    episodeWallHitEpisodeWorker = false;
    actionHistoryBufferEpisodeWorker = [];
    previousDistanceToClosestCircleEpisodeWorker = undefined;
    stagnationTrackingStepsEpisodeWorker = 0;
}

function getCurrentStateWorker() {
    let closestCircle = null;
    let minDistanceSquared = Infinity;
    if (circlesWorker.length === 0) generateCircleWorker();

    circlesWorker.forEach(circle => {
        const dx = squareXWorker + SQUARE_SIZE_WORKER / 2 - circle.x;
        const dy = squareYWorker + SQUARE_SIZE_WORKER / 2 - circle.y;
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            closestCircle = circle;
        }
    });
    const minDistance = Math.sqrt(minDistanceSquared);
    const stateFeatures = [
        squareXWorker / canvasWidthWorker,
        squareYWorker / canvasHeightWorker,
        (closestCircle ? closestCircle.x : 0) / canvasWidthWorker,
        (closestCircle ? closestCircle.y : 0) / canvasHeightWorker,
        (closestCircle ? minDistance : 0) / Math.sqrt(canvasWidthWorker * canvasWidthWorker + canvasHeightWorker * canvasHeightWorker)
    ];
    return { stateFeatures, minDistance, closestCircle, minDistanceSquared };
}

function stepEnvironmentAndGetRewardWorker(actionIndex, timeStep, currentStateData) {
    let currentReward = envConfigWorker.PENALTY_PER_STEP || -0.01;
    let collectedCount = 0;

    // ... (La logique complète de stepEnvironmentAndGetRewardWorker de votre projet original va ici)
		// C'est une version simplifiée pour la lisibilité. Assurez-vous d'utiliser la vôtre.
		let nextSquareX = squareXWorker;
    let nextSquareY = squareYWorker;

    switch (actionIndex) {
        case 0: nextSquareY -= SQUARE_SPEED_WORKER; break;
        case 1: nextSquareY += SQUARE_SPEED_WORKER; break;
        case 2: nextSquareX -= SQUARE_SPEED_WORKER; break;
        case 3: nextSquareX += SQUARE_SPEED_WORKER; break;
        case 4: nextSquareY -= SQUARE_SPEED_WORKER; nextSquareX -= SQUARE_SPEED_WORKER; break;
        case 5: nextSquareY -= SQUARE_SPEED_WORKER; nextSquareX += SQUARE_SPEED_WORKER; break;
        case 6: nextSquareY += SQUARE_SPEED_WORKER; nextSquareX -= SQUARE_SPEED_WORKER; break;
        case 7: nextSquareY += SQUARE_SPEED_WORKER; nextSquareX += SQUARE_SPEED_WORKER; break;
    }
		
		squareXWorker = Math.max(0, Math.min(canvasWidthWorker - SQUARE_SIZE_WORKER, nextSquareX));
    squareYWorker = Math.max(0, Math.min(canvasHeightWorker - SQUARE_SIZE_WORKER, nextSquareY));

    // Vérifier collection
    for (let i = circlesWorker.length - 1; i >= 0; i--) {
        const circle = circlesWorker[i];
        const dx = (squareXWorker + SQUARE_SIZE_WORKER / 2) - circle.x;
        const dy = (squareYWorker + SQUARE_SIZE_WORKER / 2) - circle.y;
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < ((CIRCLE_RADIUS_WORKER + SQUARE_SIZE_WORKER / 2 + (envConfigWorker.COLLECTION_DISTANCE_THRESHOLD || 10)) ** 2)) {
            circlesWorker.splice(i, 1);
            currentReward += envConfigWorker.REWARD_COLLECT_CIRCLE || 10;
            collectedCount++;
            if (circlesWorker.length < MAX_CIRCLES_WORKER) {
                generateCircleWorker();
            }
        }
    }

    const { stateFeatures: nextStateFeatures } = getCurrentStateWorker();
    return { reward: currentReward, collectedCount: collectedCount, nextStateFeatures: nextStateFeatures };
}


async function evaluateIndividualWorker(agent, simulationParams) {
	let totalReward = 0;
	let totalCirclesCollected = 0;
	const MAX_TIME_STEPS_PER_EPISODE = 300; // Doit être cohérent avec le thread principal

	for (let i = 0; i < simulationParams.numEpisodes; i++) {
		resetGameEnvironmentWorker();
		let episodeReward = 0;
		let episodeCirclesCollected = 0;
		let timeStep = 0;

		let currentStateData = getCurrentStateWorker();
		let currentState = currentStateData.stateFeatures;

		while (timeStep < MAX_TIME_STEPS_PER_EPISODE) {
			timeStep++;
			const actionIndex = agent.selectAction(currentState);
			const {
				reward: stepReward,
				collectedCount: stepCollectedCount,
				nextStateFeatures: nextState
			} = stepEnvironmentAndGetRewardWorker(actionIndex, timeStep, currentStateData);

			episodeReward += stepReward;
			episodeCirclesCollected += stepCollectedCount;

			if (simulationParams.isTrainingRun) {
				const done = (timeStep >= MAX_TIME_STEPS_PER_EPISODE || circlesWorker.length === 0);
				agent.train(currentState, actionIndex, stepReward, nextState, done);
			}
			currentState = nextState;
			currentStateData = getCurrentStateWorker(); // Mettre à jour les données complètes pour le prochain pas

			if (circlesWorker.length === 0) break;
		}
		totalReward += episodeReward;
		totalCirclesCollected += episodeCirclesCollected;
	}
	return {
		totalReward,
		totalCirclesCollected
	};
}


// --- Gestionnaire de messages principal du Worker ---
self.onmessage = async function(e) {
	const {
		type,
		data
	} = e.data;

	if (type === 'evolve') {
		const newPopulationBrains = evolvePopulation(data);
		self.postMessage({
			type: 'evolutionComplete',
			data: {
				newPopulationBrains
			},
		});

	} else if (type === 'evaluate') {
		const {
			agentBrain,
			simulationParams,
			originalIndex
		} = data;

		// Initialiser l'environnement du worker avec les paramètres reçus
		envConfigWorker = simulationParams.envConfig;
		canvasWidthWorker = envConfigWorker.canvasWidth;
		canvasHeightWorker = envConfigWorker.canvasHeight;
		SQUARE_SIZE_WORKER = envConfigWorker.SQUARE_SIZE;
		SQUARE_SPEED_WORKER = envConfigWorker.SQUARE_SPEED;
		CIRCLE_RADIUS_WORKER = envConfigWorker.CIRCLE_RADIUS;
		MAX_CIRCLES_WORKER = envConfigWorker.MAX_CIRCLES;


		const agent = new ActorCritic(
			simulationParams.learningRateActor,
			simulationParams.learningRateCritic,
			simulationParams.gamma,
			simulationParams.numStates,
			simulationParams.numActions
		);
		agent.loadBrain(agentBrain);

		const {
			totalReward: fitness,
			totalCirclesCollected: circlesCollectedByIndividual
		} = await evaluateIndividualWorker(agent, simulationParams);

		// Renvoyer le résultat avec le cerveau mis à jour par l'entraînement
		self.postMessage({
			type: 'evaluationComplete',
			data: {
				fitness,
				agentBrain: agent.saveBrain(),
				circlesCollectedByIndividual,
				originalIndex,
			}
		});
	}
};