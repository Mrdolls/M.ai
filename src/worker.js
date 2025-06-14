// worker.js
// Ce script s'exécute dans un Web Worker et contient la logique de simulation pour un individu.

// Import de la classe ActorCritic depuis son propre fichier
import { ActorCritic } from './actorCritic.js';

// Déclarations de variables globales pour l'environnement simulé dans le worker
let squareXWorker;
let squareYWorker;
let circlesWorker; // Renommé pour éviter toute confusion avec le thread principal
let canvasWidthWorker;
let canvasHeightWorker;
let SQUARE_SIZE_WORKER;
let SQUARE_SPEED_WORKER;
let CIRCLE_RADIUS_WORKER;
let MAX_CIRCLES_WORKER;

// Paramètres de récompense/pénalité, passés par le thread principal
let REWARD_COLLECT_CIRCLE_WORKER;
let PENALTY_PER_STEP_WORKER;
let BONUS_QUICK_COLLECTION_BASE_WORKER;
let BONUS_QUICK_COLLECTION_DECAY_WORKER;
let BONUS_PROXIMITY_WORKER;
let PROXIMITY_THRESHOLD_WORKER;
let BONUS_ACTION_VARIATION_PER_STEP_WORKER;
let ACTION_HISTORY_BUFFER_SIZE_WORKER;
let MIN_UNIQUE_ACTIONS_FOR_BONUS_WORKER;
let BONUS_NO_WALL_HIT_EPISODE_WORKER;
let PENALTY_WALL_HIT_PER_STEP_WORKER;
let PENALTY_INACTIVITY_WORKER;
let INACTIVITY_THRESHOLD_STEPS_WORKER;
let PENALTY_REPETITIVE_ACTION_WORKER;
let REPETITIVE_ACTION_THRESHOLD_STEPS_WORKER;
let PENALTY_EARLY_EMPTY_EPISODE_WORKER;
let EARLY_END_TIME_THRESHOLD_WORKER;
let COLLECTION_DISTANCE_THRESHOLD_WORKER;
let PENALTY_STAGNATION_WORKER;
let STAGNATION_THRESHOLD_STEPS_WORKER;
let PENALTY_ACTION_CHANGE_WORKER;

// Variables d'état pour le calcul de récompense par épisode dans le worker
let lastSquareXEpisodeWorker;
let lastSquareYEpisodeWorker;
let inactivityStepsEpisodeWorker;
let lastActionIndexEpisodeWorker = -1;
let repetitiveActionStepsEpisodeWorker;
let episodeWallHitEpisodeWorker;
let actionHistoryBufferEpisodeWorker = [];
let previousDistanceToClosestCircleEpisodeWorker;
let stagnationTrackingStepsEpisodeWorker = 0;


/**
 * Renvoie un entier aléatoire entre min et max (inclus).
 */
function getRandomIntWorker(min, max) {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Génère un nouveau cercle à une position aléatoire sans chevaucher le carré ou les autres cercles.
 */
function generateCircleWorker() {
	let newCircleX, newCircleY;
	let collision;
	console.log('Worker: Generating circle...'); // Added logging
	let attempts = 0;
	const MAX_ATTEMPTS = 100; // Prevent infinite loop for circle generation
	do {
		collision = false;
		newCircleX = getRandomIntWorker(CIRCLE_RADIUS_WORKER, canvasWidthWorker - CIRCLE_RADIUS_WORKER);
		newCircleY = getRandomIntWorker(CIRCLE_RADIUS_WORKER, canvasHeightWorker - CIRCLE_RADIUS_WORKER);
		// Vérifier la collision avec le carré
		if (newCircleX + SQUARE_SIZE_WORKER > squareXWorker && newCircleX - CIRCLE_RADIUS_WORKER < squareXWorker + SQUARE_SIZE_WORKER &&
			newCircleY + CIRCLE_RADIUS_WORKER > squareYWorker && newCircleY - CIRCLE_RADIUS_WORKER < squareYWorker + SQUARE_SIZE_WORKER) { // Corrected variable name
			collision = true;
		}
		// Vérifier la collision avec les cercles existants
		for (const existingCircle of circlesWorker) {
			const dx = newCircleX - existingCircle.x;
			const dy = newCircleY - existingCircle.y;
			const distanceSquared = dx * dx + dy * dy;
			if (distanceSquared < (CIRCLE_RADIUS_WORKER * 2) * (CIRCLE_RADIUS_WORKER * 2)) {
				collision = true;
				break;
			}
		}
		attempts++;
		if (attempts > MAX_ATTEMPTS) {
			console.warn("Worker: Max attempts reached for circle generation, might place overlapping circle.");
			break; // Break to prevent infinite loop, even if overlap occurs
		}
	} while (collision);
	circlesWorker.push({ x: newCircleX, y: newCircleY });
	console.log('Worker: Circle generated at', newCircleX, newCircleY); // Added logging
}

/**
 * Réinitialise l'environnement de jeu simulé.
 */
function resetGameEnvironmentWorker() {
	console.log('Worker: Resetting environment...'); // Added logging
	squareXWorker = canvasWidthWorker / 2 - SQUARE_SIZE_WORKER / 2;
	squareYWorker = canvasHeightWorker / 2 - SQUARE_SIZE_WORKER / 2;
	circlesWorker = [];
	for (let i = 0; i < MAX_CIRCLES_WORKER; i++) {
		generateCircleWorker();
	}

	// Réinitialisation des variables d'état pour les récompenses spécifiques au worker
	lastSquareXEpisodeWorker = squareXWorker;
	lastSquareYEpisodeWorker = squareYWorker;
	inactivityStepsEpisodeWorker = 0;
	lastActionIndexEpisodeWorker = -1;
	repetitiveActionStepsEpisodeWorker = 0;
	episodeWallHitEpisodeWorker = false;
	actionHistoryBufferEpisodeWorker = [];
	previousDistanceToClosestCircleEpisodeWorker = undefined;
	stagnationTrackingStepsEpisodeWorker = 0;
	console.log('Worker: Environment reset complete.'); // Added logging
}

/**
 * Calcule l'état actuel de l'environnement simulé.
 * @returns {object} Un objet contenant les caractéristiques de l'état, la distance minimale et le cercle le plus proche.
 */
function getCurrentStateWorker() {
	let closestCircle = null;
	let minDistanceSquared = Infinity;

	if (circlesWorker.length === 0) {
		console.warn('Worker: No circles found, generating a new one in getCurrentStateWorker.'); // Added warning
		generateCircleWorker();
	}

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

	// Normalisation des valeurs de l'état entre 0 et 1
	const stateFeatures = [
		squareXWorker / canvasWidthWorker,
		squareYWorker / canvasHeightWorker,
		(closestCircle ? closestCircle.x : 0) / canvasWidthWorker,
		(closestCircle ? closestCircle.y : 0) / canvasHeightWorker,
		(closestCircle ? minDistance : 0) / Math.sqrt(canvasWidthWorker * canvasWidthWorker + canvasHeightWorker * canvasHeightWorker)
	];
	return { stateFeatures, minDistance, closestCircle, minDistanceSquared };
}

/**
 * Effectue un pas dans l'environnement simulé, met à jour la position du carré,
 * vérifie les collisions et calcule la récompense pour ce pas.
 * @param {number} actionIndex - L'action choisie par l'agent.
 * @param {number} timeStep - Le numéro du pas de temps actuel dans l'épisode.
 * @param {object} currentStateData - Les données de l'état actuel (stateFeatures, minDistance, minDistanceSquared, closestCircle) obtenues de getCurrentStateWorker().
 * @returns {object} Un objet contenant la récompense pour ce pas, le nombre de ronds collectés et les fonctionnalités de l'état suivant.
 */
function stepEnvironmentAndGetRewardWorker(actionIndex, timeStep, currentStateData) {
	let currentReward = PENALTY_PER_STEP_WORKER;
	let collectedCount = 0;

	const { minDistance: currentDistanceToClosest } = currentStateData;

	if (previousDistanceToClosestCircleEpisodeWorker === undefined) {
		previousDistanceToClosestCircleEpisodeWorker = currentDistanceToClosest;
	}

	if (lastActionIndexEpisodeWorker !== -1 && actionIndex !== lastActionIndexEpisodeWorker) {
		currentReward += PENALTY_ACTION_CHANGE_WORKER;
	}

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

	const willHitWall = (nextSquareX < 0 || nextSquareX > canvasWidthWorker - SQUARE_SIZE_WORKER ||
		nextSquareY < 0 || nextSquareY > canvasHeightWorker - SQUARE_SIZE_WORKER);
	if (willHitWall) {
		currentReward += PENALTY_WALL_HIT_PER_STEP_WORKER;
		episodeWallHitEpisodeWorker = true;
	}

	squareXWorker = Math.max(0, Math.min(canvasWidthWorker - SQUARE_SIZE_WORKER, nextSquareX));
	squareYWorker = Math.max(0, Math.min(canvasHeightWorker - SQUARE_SIZE_WORKER, nextSquareY));

	if (squareXWorker === lastSquareXEpisodeWorker && squareYWorker === lastSquareYEpisodeWorker) {
		inactivityStepsEpisodeWorker++;
	} else {
		inactivityStepsEpisodeWorker = 0;
	}
	lastSquareXEpisodeWorker = squareXWorker;
	lastSquareYEpisodeWorker = squareYWorker;

	if (inactivityStepsEpisodeWorker > 0 && currentDistanceToClosest > (CIRCLE_RADIUS_WORKER + SQUARE_SIZE_WORKER / 2) && currentDistanceToClosest < PROXIMITY_THRESHOLD_WORKER) {
		stagnationTrackingStepsEpisodeWorker++;
		if (stagnationTrackingStepsEpisodeWorker >= STAGNATION_THRESHOLD_STEPS_WORKER) {
			currentReward += PENALTY_STAGNATION_WORKER;
		}
	} else {
		stagnationTrackingStepsEpisodeWorker = 0;
	}

	const previousCollectedCount = collectedCount;
	if (actionIndex === lastActionIndexEpisodeWorker && previousCollectedCount === 0) {
		repetitiveActionStepsEpisodeWorker++;
		if (repetitiveActionStepsEpisodeWorker >= REPETITIVE_ACTION_THRESHOLD_STEPS_WORKER) {
			currentReward += PENALTY_REPETITIVE_ACTION_WORKER;
		}
	} else {
		repetitiveActionStepsEpisodeWorker = 0;
	}
	lastActionIndexEpisodeWorker = actionIndex;

	actionHistoryBufferEpisodeWorker.push(actionIndex);
	if (actionHistoryBufferEpisodeWorker.length > ACTION_HISTORY_BUFFER_SIZE_WORKER) {
		actionHistoryBufferEpisodeWorker.shift();
	}
	const uniqueActions = new Set(actionHistoryBufferEpisodeWorker).size;
	if (uniqueActions >= MIN_UNIQUE_ACTIONS_FOR_BONUS_WORKER) {
		currentReward += BONUS_ACTION_VARIATION_PER_STEP_WORKER;
	}

	for (let i = circlesWorker.length - 1; i >= 0; i--) {
		const circle = circlesWorker[i];
		const dx = (squareXWorker + SQUARE_SIZE_WORKER / 2) - circle.x;
		const dy = (squareYWorker + SQUARE_SIZE_WORKER / 2) - circle.y;
		const distanceSquared = dx * dx + dy * dy;

		if (distanceSquared < ((CIRCLE_RADIUS_WORKER + SQUARE_SIZE_WORKER / 2 + COLLECTION_DISTANCE_THRESHOLD_WORKER) * (CIRCLE_RADIUS_WORKER + SQUARE_SIZE_WORKER / 2 + COLLECTION_DISTANCE_THRESHOLD_WORKER))) {
			circlesWorker.splice(i, 1);
			currentReward += REWARD_COLLECT_CIRCLE_WORKER;
			collectedCount++;

			currentReward += Math.max(0, BONUS_QUICK_COLLECTION_BASE_WORKER - (timeStep * BONUS_QUICK_COLLECTION_DECAY_WORKER));

			if (circlesWorker.length < MAX_CIRCLES_WORKER) {
				generateCircleWorker();
			}
		}
	}

	const { minDistance: newDistanceToClosest } = getCurrentStateWorker();
	if (newDistanceToClosest < previousDistanceToClosestCircleEpisodeWorker && newDistanceToClosest > 0 && newDistanceToClosest < PROXIMITY_THRESHOLD_WORKER) {
		currentReward += BONUS_PROXIMITY_WORKER;
	}
	previousDistanceToClosestCircleEpisodeWorker = newDistanceToClosest;

	const { stateFeatures: nextStateFeatures } = getCurrentStateWorker();

	return { reward: currentReward, collectedCount: collectedCount, nextStateFeatures: nextStateFeatures };
}

/**
 * Évalue la performance d'un individu (agent ActorCritic) sur un nombre donné d'épisodes.
 * @param {ActorCritic} agent - L'instance de l'agent à évaluer.
 * @param {number} numEpisodes - Le nombre d'épisodes pour l'évaluation.
 * @param {boolean} isTrainingRun - Vrai si c'est un run d'entraînement (avec apprentissage), faux sinon.
 * @returns {object} Un objet contenant la récompense totale accumulée et les ronds ramassés.
 */
async function evaluateIndividualWorker(agent, numEpisodes, isTrainingRun = true) {
	let totalReward = 0;
	let totalCirclesCollected = 0;
	const MAX_TIME_STEPS_PER_EPISODE = 300;

	console.log(`Worker: Evaluating individual for ${numEpisodes} episodes.`); // Added logging

	for (let i = 0; i < numEpisodes; i++) {
		console.log(`Worker: Starting episode ${i + 1}/${numEpisodes}.`); // Added logging
		try {
			resetGameEnvironmentWorker();
			let episodeReward = 0;
			let episodeCirclesCollected = 0;
			let timeStep = 0;
			let episodeTerminatedEarlyWithoutCollection = false;

			let currentStateData = getCurrentStateWorker();
			let currentState = currentStateData.stateFeatures;

			while (timeStep < MAX_TIME_STEPS_PER_EPISODE) {
				timeStep++;
				const actionIndex = agent.selectAction(currentState);

				const { reward: stepReward, collectedCount: stepCollectedCount, nextStateFeatures: nextState } = stepEnvironmentAndGetRewardWorker(actionIndex, timeStep, currentStateData);

				episodeReward += stepReward;
				episodeCirclesCollected += stepCollectedCount;

				if (isTrainingRun) {
					const done = (timeStep >= MAX_TIME_STEPS_PER_EPISODE || circlesWorker.length === 0);
					agent.train(currentState, actionIndex, stepReward, nextState, done);
				}

				currentState = nextState;
				currentStateData = getCurrentStateWorker();

				if (circlesWorker.length === 0 || timeStep >= MAX_TIME_STEPS_PER_EPISODE) {
					episodeTerminatedEarlyWithoutCollection = (episodeCirclesCollected === 0 && timeStep < EARLY_END_TIME_THRESHOLD_WORKER);
					break;
				}
			}
			totalReward += episodeReward;
			totalCirclesCollected += episodeCirclesCollected;

			if (!episodeWallHitEpisodeWorker) {
				totalReward += BONUS_NO_WALL_HIT_EPISODE_WORKER;
			}

			if (episodeTerminatedEarlyWithoutCollection) {
				totalReward += PENALTY_EARLY_EMPTY_EPISODE_WORKER;
			}
			console.log(`Worker: Episode ${i + 1} finished. Reward: ${episodeReward.toFixed(2)}, Circles: ${episodeCirclesCollected}`); // Added logging

		} catch (error) {
			console.error(`Worker: Error in evaluateIndividualWorker for episode ${i + 1}:`, error);
			// If an episode fails, penalize this individual heavily
			totalReward = -Infinity;
			totalCirclesCollected = 0;
			break; // Stop evaluating further episodes for this individual
		}
	}
	console.log(`Worker: Individual evaluation complete. Total Reward: ${totalReward.toFixed(2)}, Total Circles: ${totalCirclesCollected}`); // Added logging
	return { totalReward, totalCirclesCollected };
}

// Écoute les messages du thread principal
self.onmessage = async function (e) {
	console.log('Worker: Message received from main thread.'); // Added logging
	const { agentBrain, simulationParams } = e.data;

	// Initialiser les paramètres d'environnement pour le worker
	canvasWidthWorker = simulationParams.envConfig.canvasWidth;
	canvasHeightWorker = simulationParams.envConfig.canvasHeight;
	SQUARE_SIZE_WORKER = simulationParams.envConfig.SQUARE_SIZE;
	SQUARE_SPEED_WORKER = simulationParams.envConfig.SQUARE_SPEED;
	CIRCLE_RADIUS_WORKER = simulationParams.envConfig.CIRCLE_RADIUS;
	MAX_CIRCLES_WORKER = simulationParams.envConfig.MAX_CIRCLES;

	REWARD_COLLECT_CIRCLE_WORKER = simulationParams.envConfig.REWARD_COLLECT_CIRCLE;
	PENALTY_PER_STEP_WORKER = simulationParams.envConfig.PENALTY_PER_STEP;
	BONUS_QUICK_COLLECTION_BASE_WORKER = simulationParams.envConfig.BONUS_QUICK_COLLECTION_BASE;
	BONUS_QUICK_COLLECTION_DECAY_WORKER = simulationParams.envConfig.BONUS_QUICK_COLLECTION_DECAY;
	BONUS_PROXIMITY_WORKER = simulationParams.envConfig.BONUS_PROXIMITY;
	PROXIMITY_THRESHOLD_WORKER = simulationParams.envConfig.PROXIMITY_THRESHOLD;
	BONUS_ACTION_VARIATION_PER_STEP_WORKER = simulationParams.envConfig.BONUS_ACTION_VARIATION_PER_STEP;
	ACTION_HISTORY_BUFFER_SIZE_WORKER = simulationParams.envConfig.ACTION_HISTORY_BUFFER_SIZE;
	MIN_UNIQUE_ACTIONS_FOR_BONUS_WORKER = simulationParams.envConfig.MIN_UNIQUE_ACTIONS_FOR_BONUS;
	BONUS_NO_WALL_HIT_EPISODE_WORKER = simulationParams.envConfig.BONUS_NO_WALL_HIT_EPISODE;
	PENALTY_WALL_HIT_PER_STEP_WORKER = simulationParams.envConfig.PENALTY_WALL_HIT_PER_STEP;
	PENALTY_REPETITIVE_ACTION_WORKER = simulationParams.envConfig.PENALTY_REPETITIVE_ACTION;
	REPETITIVE_ACTION_THRESHOLD_STEPS_WORKER = simulationParams.envConfig.REPETITIVE_ACTION_THRESHOLD_STEPS;
	PENALTY_EARLY_EMPTY_EPISODE_WORKER = simulationParams.envConfig.PENALTY_EARLY_EMPTY_EPISODE;
	EARLY_END_TIME_THRESHOLD_WORKER = simulationParams.envConfig.EARLY_END_TIME_THRESHOLD;
	COLLECTION_DISTANCE_THRESHOLD_WORKER = simulationParams.envConfig.COLLECTION_DISTANCE_THRESHOLD;
	PENALTY_STAGNATION_WORKER = simulationParams.envConfig.PENALTY_STAGNATION;
	STAGNATION_THRESHOLD_STEPS_WORKER = simulationParams.envConfig.STAGNATION_THRESHOLD_STEPS;
	PENALTY_ACTION_CHANGE_WORKER = simulationParams.envConfig.PENALTY_ACTION_CHANGE;


	console.log('Worker: Env config loaded.'); // Added logging

	const agent = new ActorCritic(
		simulationParams.learningRateActor,
		simulationParams.learningRateCritic,
		simulationParams.gamma,
		simulationParams.numStates,
		simulationParams.numActions
	);
	agent.loadBrain(agentBrain);
	console.log('Worker: Agent loaded, starting evaluation.'); // Added logging

	const { totalReward: fitness, totalCirclesCollected: circlesCollectedByIndividual } = await evaluateIndividualWorker(agent, simulationParams.numEpisodes, simulationParams.isTrainingRun);

	console.log('Worker: Evaluation complete, posting message back.'); // Added logging
	self.postMessage({ fitness: fitness, agentBrain: agent.saveBrain(), circlesCollectedByIndividual: circlesCollectedByIndividual });
};
