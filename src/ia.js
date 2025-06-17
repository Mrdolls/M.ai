// ia.js - Implémentation de l'IA Actor-Critic (avec support génétique)
// et logique principale du jeu, refactorisé avec Web Workers.

import {
	ActorCritic
} from './actorCritic.js';

// --- Déclaration des variables globales pour le DOM, le jeu et l'entraînement ---
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const numIndividualsInput = document.getElementById('numIndividuals');
const numGenerationsInput = document.getElementById('numGenerations');
const infiniteGenerationsCheckbox = document.getElementById('infiniteGenerationsCheckbox');
const episodesPerIndividualInput = document.getElementById('episodesPerIndividual');
const mutationRateInput = document.getElementById('mutationRate');
const learningRateActorInput = document.getElementById('learningRateActor');
const learningRateCriticInput = document.getElementById('learningRateCritic');
const gammaInput = document.getElementById('gamma');
const numElitismInput = document.getElementById('numElitism');

// Références DOM pour les paramètres de récompense/pénalité
const collectionDistanceThresholdInput = document.getElementById('collectionDistanceThreshold');
const stagnationPenaltyInput = document.getElementById('stagnationPenalty');
const stagnationThresholdStepsInput = document.getElementById('stagnationThresholdSteps');
const actionChangePenaltyInput = document.getElementById('actionChangePenalty');

const currentGenerationSpan = document.getElementById('currentGeneration');
const bestRewardAllTimeSpan = document.getElementById('bestRewardAllTime');
const bestCirclesAllTimeSpan = document.getElementById('bestCirclesAllTimeSpan');

const testBestIAButton = document.getElementById('testBestIAButton');
const stopTestButtonVisual = document.getElementById('stopTestButtonVisual');
const currentTestScoreSpan = document.getElementById('currentTestScore');
const circlesCollectedTestSpan = document.getElementById('circlesCollectedTest');

const settingsPanel = document.getElementById('settingsPanel');
const settingsButton = document.getElementById('settingsButton');

const saveButton = document.getElementById('saveButton');
const loadButton = document.getElementById('loadButton');
const loadFileInput = document.getElementById('loadFileInput');
const resetButton = document.getElementById('resetButton');

const customConfirmModal = document.getElementById('customConfirmModal');
const modalMessage = document.getElementById('modalMessage');
const modalConfirmBtn = document.getElementById('modalConfirmBtn');
const modalCancelBtn = document.getElementById('modalCancelBtn');

const learningCurveChartDiv = document.getElementById('learningCurveChart');

const loaderOverlay = document.querySelector('#loader-overlay');
const closeButton = document.querySelector('#closeButton');

// --- Paramètres du Jeu et de l'IA ---
const SQUARE_SIZE = 30;
let squareX = canvas.width / 2 - SQUARE_SIZE / 2;
let squareY = canvas.height / 2 - SQUARE_SIZE / 2;
const SQUARE_SPEED = 5;
const CIRCLE_RADIUS = 10;
const MAX_CIRCLES = 5;
let circles = [];
let stateFeatureSize = 0;

// --- Variables d'état ---
let isTraining = false;
let isTestingVisual = false;
let animationFrameId = null;
let currentGeneration = 0;
let bestRewardAllTime = 0;
let bestCirclesAllTime = 0;
let learningCurveData = [];

let bestAgentInstance = null; // Stocke une instance ActorCritic complète du meilleur agent
let isCanvasRenderingEnabled = true;
let currentPopulationBrains = []; // Stocke uniquement les poids de la population
let currentGaParams = {};
let generationResults = [];
let tasksPending = 0;
let generationPassed = 0;

// --- Initialisation des Web Workers ---
const NUM_WORKERS = Math.max(2, Math.min(12, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 2));
const workers = [];


// --- FONCTIONS UTILITAIRES ---

function log(message) {
	console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
}

// --- NOUVELLE SECTION : GESTION INDEXEDDB ---

const DB_NAME = 'iaTrainingDB';
const STORE_NAME = 'bestAgentStore';

/**
 * Ouvre la base de données IndexedDB et crée l'object store si nécessaire.
 * @returns {Promise<IDBDatabase>} Une promesse qui se résout avec l'objet de la base de données.
 */
function openDB() {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, 1);
		request.onerror = (event) => reject("Erreur d'ouverture IndexedDB.");
		request.onsuccess = (event) => resolve(event.target.result);
		request.onupgradeneeded = (event) => {
			const db = event.target.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME);
			}
		};
	});
}

/**
 * Sauvegarde les données de l'agent dans IndexedDB.
 * @param {object} data - Les données de l'agent à sauvegarder.
 */
async function saveAgentToDB(data) {
	try {
		const db = await openDB();
		const transaction = db.transaction(STORE_NAME, 'readwrite');
		const store = transaction.objectStore(STORE_NAME);
		store.put(data, 'bestAgent'); // Utilise une clé statique pour toujours écraser le même enregistrement
		log("Agent sauvegardé automatiquement dans IndexedDB.");
	} catch (error) {
		console.error("Échec de la sauvegarde dans IndexedDB:", error);
	}
}

/**
 * Charge les données de l'agent depuis IndexedDB.
 * @returns {Promise<object|null>} Une promesse qui se résout avec les données ou null.
 */
async function loadAgentFromDB() {
	try {
		const db = await openDB();
		const transaction = db.transaction(STORE_NAME, 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const request = store.get('bestAgent');
		return new Promise((resolve) => {
			request.onsuccess = () => {
				resolve(request.result || null);
			};
			request.onerror = () => {
				console.error("Erreur de chargement depuis IndexedDB.");
				resolve(null);
			};
		});
	} catch (error) {
		console.error("Échec du chargement depuis IndexedDB:", error);
		return null;
	}
}

/**
 * Vide l'object store dans IndexedDB.
 */
async function clearIndexedDB() {
	try {
		const db = await openDB();
		const transaction = db.transaction(STORE_NAME, 'readwrite');
		const store = transaction.objectStore(STORE_NAME);
		store.clear();
		log("Base de données IndexedDB vidée.");
	} catch (error) {
		console.error("Échec de la suppression des données IndexedDB:", error);
	}
}

// --- FIN DE LA SECTION INDEXEDDB ---


function showCustomConfirm(message) {
	return new Promise((resolve) => {
		modalMessage.textContent = message;
		customConfirmModal.style.display = 'flex';
		const onConfirm = () => {
			customConfirmModal.style.display = 'none';
			modalConfirmBtn.removeEventListener('click', onConfirm);
			modalCancelBtn.removeEventListener('click', onCancel);
			resolve(true);
		};
		const onCancel = () => {
			customConfirmModal.style.display = 'none';
			modalConfirmBtn.removeEventListener('click', onConfirm);
			modalCancelBtn.removeEventListener('click', onCancel);
			resolve(false);
		};
		modalConfirmBtn.addEventListener('click', onConfirm);
		modalCancelBtn.addEventListener('click', onCancel);
	});
}

function updateRewardParameters() {
	// Cette fonction met à jour les variables de récompense globales à partir de l'UI
	// Pour la concision, le corps de la fonction est omis, mais il est nécessaire pour votre logique
	log("Paramètres de récompense mis à jour.");
}


// --- GESTION DES WORKERS ---

function initializeWorkers() {
	if (workers.length > 0) return;
	log(`Initialisation de ${NUM_WORKERS} Web Workers...`);
	for (let i = 0; i < NUM_WORKERS; i++) {
		const worker = new Worker('./src/worker.js', {
			type: 'module'
		});
		worker.onmessage = onWorkerMessage;
		worker.onerror = (e) => console.error(`Erreur du Worker ${i}:`, e.message, e.filename, e.lineno);
		workers.push(worker);
	}
}

function onWorkerMessage(e) {
	const {
		type,
		data
	} = e.data;

	if (type === 'evaluationComplete') {
		generationResults[data.originalIndex] = data;
		tasksPending--;
		if (tasksPending === 0) {
			processGenerationResults(generationResults, currentGaParams);
		}
	} else if (type === 'evolutionComplete') {
		currentPopulationBrains = data.newPopulationBrains;
		currentGeneration++;
		generationPassed++;
		runSingleGeneration(currentPopulationBrains, currentGaParams);
	}
}


// --- LOGIQUE DE L'ALGORITHME GÉNÉTIQUE ---

async function runGenerations() {
	log("Démarrage de l'entraînement génétique...");
	isTraining = true;
	startButton.disabled = true;
	stopButton.disabled = false;
	testBestIAButton.disabled = true;
	isCanvasRenderingEnabled = false;
	generationPassed = 1; // On passe à la générationPassed 1
	if (stateFeatureSize === 0) {
		resetGameEnvironment();
		stateFeatureSize = getCurrentState().stateFeatures.length;
		log(`Taille des caractéristiques d'état déterminée : ${stateFeatureSize}`);
	}

	currentGaParams = {
		numIndividuals: parseInt(numIndividualsInput.value),
		numGenerations: parseInt(numGenerationsInput.value),
		episodesPerIndividual: parseInt(episodesPerIndividualInput.value),
		mutationRate: parseFloat(mutationRateInput.value),
		learningRateActor: parseFloat(learningRateActorInput.value),
		learningRateCritic: parseFloat(learningRateCriticInput.value),
		gamma: parseFloat(gammaInput.value),
		numElitism: parseInt(numElitismInput.value),
		isInfiniteMode: infiniteGenerationsCheckbox.checked,
		numActions: 8,
		numStates: stateFeatureSize,
	};

	currentPopulationBrains = [];
	if (bestAgentInstance) {
		log("Initialisation de la population à partir du meilleur agent existant.");
		currentPopulationBrains.push(bestAgentInstance.saveBrain());
		for (let i = 1; i < currentGaParams.numIndividuals; i++) {
			const agent = new ActorCritic(currentGaParams.learningRateActor, currentGaParams.learningRateCritic, currentGaParams.gamma, currentGaParams.numStates, currentGaParams.numActions, bestAgentInstance.getActorWeights(), bestAgentInstance.getCriticWeights());
			agent.setActorWeights(mutate(agent.getActorWeights(), 0.1));
			agent.setCriticWeights(mutate(agent.getCriticWeights(), 0.1));
			currentPopulationBrains.push(agent.saveBrain());
		}
	} else {
		log("Initialisation d'une nouvelle population aléatoire.");
		for (let i = 0; i < currentGaParams.numIndividuals; i++) {
			const agent = new ActorCritic(currentGaParams.learningRateActor, currentGaParams.learningRateCritic, currentGaParams.gamma, currentGaParams.numStates, currentGaParams.numActions);
			currentPopulationBrains.push(agent.saveBrain());
		}
	}

	if (currentGeneration === 0) {
		log("Initialisation d'une nouvelle session d'entraînement.");
		if (stateFeatureSize === 0) {
			resetGameEnvironment();
			stateFeatureSize = getCurrentState().stateFeatures.length;
			log(`Taille des caractéristiques d'état déterminée : ${stateFeatureSize}`);
		}

		currentGaParams.numStates = stateFeatureSize; // S'assurer que le paramètre est à jour
		currentPopulationBrains = [];
		if (bestAgentInstance) {
			log("Initialisation de la population à partir du meilleur agent existant.");
			currentPopulationBrains.push(bestAgentInstance.saveBrain());
			for (let i = 1; i < currentGaParams.numIndividuals; i++) {
				const agent = new ActorCritic(currentGaParams.learningRateActor, currentGaParams.learningRateCritic, currentGaParams.gamma, currentGaParams.numStates, currentGaParams.numActions, bestAgentInstance.getActorWeights(), bestAgentInstance.getCriticWeights());
				agent.setActorWeights(mutate(agent.getActorWeights(), 0.1));
				agent.setCriticWeights(mutate(agent.getCriticWeights(), 0.1));
				currentPopulationBrains.push(agent.saveBrain());
			}
		} else {
			log("Initialisation d'une nouvelle population aléatoire.");
			for (let i = 0; i < currentGaParams.numIndividuals; i++) {
				const agent = new ActorCritic(currentGaParams.learningRateActor, currentGaParams.learningRateCritic, currentGaParams.gamma, currentGaParams.numStates, currentGaParams.numActions);
				currentPopulationBrains.push(agent.saveBrain());
			}
		}

		currentGeneration = 1; // On passe à la génération 1
	} else {
		log(`Reprise de l'entraînement à la génération ${currentGeneration}.`);
		// Si ce n'est pas la génération 0, on ne touche ni à la population, ni au compteur.
	}

	runSingleGeneration(currentPopulationBrains, currentGaParams);
}

function runSingleGeneration(populationBrains, gaParams) {
	if (!isTraining || (!gaParams.isInfiniteMode && generationPassed > gaParams.numGenerations)) {
		stopTraining(false);
		log("Entraînement terminé.");
		return;
	}

	log(`--- Lancement Génération ${currentGeneration} ---`);
	currentGenerationSpan.textContent = currentGeneration;

	tasksPending = populationBrains.length;
	generationResults = new Array(tasksPending);
	const simulationParams = getSimulationParameters();

	populationBrains.forEach((brain, index) => {
		const worker = workers[index % NUM_WORKERS];
		worker.postMessage({
			type: 'evaluate',
			data: {
				agentBrain: brain,
				simulationParams: simulationParams,
				originalIndex: index,
			}
		});
	});
}

function processGenerationResults(results, gaParams) {
	log("Évaluation de la génération terminée. Traitement des résultats...");

	let bestFitnessThisGen = -Infinity;
	let bestIndividualThisGen = null;
	let sumFitnessThisGen = 0;

	results.forEach(result => {
		sumFitnessThisGen += result.fitness;
		if (result.fitness > bestFitnessThisGen) {
			bestFitnessThisGen = result.fitness;
			bestIndividualThisGen = result;
		}
	});

	const averageFitnessThisGen = sumFitnessThisGen / results.length;
	updateLearningCurveChart(averageFitnessThisGen);
	log(`Récompense moyenne: ${averageFitnessThisGen.toFixed(2)}. Meilleure récompense: ${bestFitnessThisGen.toFixed(2)}`);

	if (bestFitnessThisGen > bestRewardAllTime) {
		bestRewardAllTime = bestFitnessThisGen;
		bestCirclesAllTime = bestIndividualThisGen.circlesCollectedByIndividual;
		bestRewardAllTimeSpan.textContent = bestRewardAllTime.toFixed(2);
		bestCirclesAllTimeSpan.textContent = bestCirclesAllTime;
		if (!bestAgentInstance) {
			bestAgentInstance = new ActorCritic(gaParams.learningRateActor, gaParams.learningRateCritic, gaParams.gamma, gaParams.numStates, gaParams.numActions);
			const dataToSave = prepareDataForSaving();
			saveAgentToDB(dataToSave); //

		}
		bestAgentInstance.loadBrain(bestIndividualThisGen.agentBrain);
		log(`Nouveau meilleur score global ! Récompense: ${bestRewardAllTime.toFixed(2)}`);

		// MODIFICATION : Sauvegarde automatique dans IndexedDB
		const dataToSave = prepareDataForSaving();
		saveAgentToDB(dataToSave);
	}

	log("Lancement de l'évolution de la population...");
	const evolutionWorker = workers[0];
	const populationDataForEvolution = results.map(r => ({
		agentBrain: r.agentBrain,
		fitness: r.fitness,
	}));
	evolutionWorker.postMessage({
		type: 'evolve',
		data: {
			populationData: populationDataForEvolution,
			gaParams: gaParams,
		},
	});
}

// ia.js

function stopTraining(fromUser = false) {
	isTraining = false;
	startButton.disabled = false;
	stopButton.disabled = true;
	if (bestAgentInstance) {
		testBestIAButton.disabled = false;
	}
	isCanvasRenderingEnabled = true;

	// AJOUT : Sauvegarder l'état actuel si l'arrêt est manuel et qu'un agent existe
	if (fromUser && bestAgentInstance) {
		log("Sauvegarde de l'état actuel sur arrêt manuel...");
		const dataToSave = prepareDataForSaving();
		saveAgentToDB(dataToSave);
	}

	log("Entraînement arrêté" + (fromUser ? " par l'utilisateur." : "."));
}

function getSimulationParameters() {
	return {
		numEpisodes: parseInt(episodesPerIndividualInput.value),
		isTrainingRun: true,
		learningRateActor: parseFloat(learningRateActorInput.value),
		learningRateCritic: parseFloat(learningRateCriticInput.value),
		gamma: parseFloat(gammaInput.value),
		numStates: stateFeatureSize,
		numActions: 8,
		envConfig: {
			canvasWidth: canvas.width,
			canvasHeight: canvas.height,
			SQUARE_SIZE,
			SQUARE_SPEED,
			CIRCLE_RADIUS,
			MAX_CIRCLES,
			COLLECTION_DISTANCE_THRESHOLD: parseFloat(collectionDistanceThresholdInput.value),
			// ... inclure TOUS les autres paramètres de récompense de l'UI
		}
	};
}

// --- FONCTIONS DE L'ENVIRONNEMENT DE JEU (POUR LE TEST VISUEL) ---
// Ces fonctions sont nécessaires sur le thread principal pour le test visuel
// et l'initialisation.

function mutate(weights, mutationRate, mutationStrength = 0.1) {
	const mutatedWeights = new Float32Array(weights);
	for (let i = 0; i < mutatedWeights.length; i++) {
		if (Math.random() < mutationRate) {
			mutatedWeights[i] += (Math.random() * 2 - 1) * mutationStrength;
		}
	}
	return mutatedWeights;
}

function generateCircle() {
	let newCircleX, newCircleY, collision;
	do {
		collision = false;
		newCircleX = Math.floor(Math.random() * (canvas.width - CIRCLE_RADIUS * 2)) + CIRCLE_RADIUS;
		newCircleY = Math.floor(Math.random() * (canvas.height - CIRCLE_RADIUS * 2)) + CIRCLE_RADIUS;
		for (const c of circles) {
			const dx = newCircleX - c.x;
			const dy = newCircleY - c.y;
			if (Math.sqrt(dx * dx + dy * dy) < CIRCLE_RADIUS * 2) {
				collision = true;
				break;
			}
		}
	} while (collision);
	circles.push({ x: newCircleX, y: newCircleY });
}

function resetGameEnvironment() {
	squareX = canvas.width / 2 - SQUARE_SIZE / 2;
	squareY = canvas.height / 2 - SQUARE_SIZE / 2;
	circles = [];
	for (let i = 0; i < MAX_CIRCLES; i++) {
		generateCircle();
	}
}

function getCurrentState() {
	let closestCircle = null;
	let minDistanceSquared = Infinity;
	if (circles.length === 0) generateCircle();

	circles.forEach(circle => {
		const dx = squareX + SQUARE_SIZE / 2 - circle.x;
		const dy = squareY + SQUARE_SIZE / 2 - circle.y;
		const distanceSquared = dx * dx + dy * dy;
		if (distanceSquared < minDistanceSquared) {
			minDistanceSquared = distanceSquared;
			closestCircle = circle;
		}
	});

	const minDistance = Math.sqrt(minDistanceSquared);
	const stateFeatures = [
		squareX / canvas.width,
		squareY / canvas.height,
		(closestCircle ? closestCircle.x : 0) / canvas.width,
		(closestCircle ? closestCircle.y : 0) / canvas.height,
		(closestCircle ? minDistance : 0) / Math.sqrt(canvas.width * canvas.width + canvas.height * canvas.height)
	];
	return { stateFeatures, minDistance, closestCircle };
}

function stepEnvironmentAndGetReward(actionIndex) {
	// La logique de cette fonction est nécessaire pour le test visuel.
	// Elle doit être complète, similaire à celle du worker.
	let reward = -0.01;
	let collectedCount = 0;
	let nextSquareX = squareX;
	let nextSquareY = squareY;

	switch (actionIndex) {
		case 0: nextSquareY -= SQUARE_SPEED; break;
		case 1: nextSquareY += SQUARE_SPEED; break;
		case 2: nextSquareX -= SQUARE_SPEED; break;
		case 3: nextSquareX += SQUARE_SPEED; break;
		case 4: nextSquareY -= SQUARE_SPEED; nextSquareX -= SQUARE_SPEED; break;
		case 5: nextSquareY -= SQUARE_SPEED; nextSquareX += SQUARE_SPEED; break;
		case 6: nextSquareY += SQUARE_SPEED; nextSquareX -= SQUARE_SPEED; break;
		case 7: nextSquareY += SQUARE_SPEED; nextSquareX += SQUARE_SPEED; break;
	}

	squareX = Math.max(0, Math.min(canvas.width - SQUARE_SIZE, nextSquareX));
	squareY = Math.max(0, Math.min(canvas.height - SQUARE_SIZE, nextSquareY));

	for (let i = circles.length - 1; i >= 0; i--) {
		const circle = circles[i];
		const dx = (squareX + SQUARE_SIZE / 2) - circle.x;
		const dy = (squareY + SQUARE_SIZE / 2) - circle.y;
		if (Math.sqrt(dx * dx + dy * dy) < CIRCLE_RADIUS + SQUARE_SIZE / 2) {
			circles.splice(i, 1);
			reward += 10;
			collectedCount++;
			generateCircle();
		}
	}
	return { reward, collectedCount };
}

function drawGame() {
	if (!isCanvasRenderingEnabled) return;

	ctx.clearRect(0, 0, canvas.width, canvas.height);
	// Préparation des styles pour les cercles
	ctx.shadowBlur = 10;
	ctx.shadowColor = "blue";
	ctx.fillStyle = '#4800ff';

	for (const { x, y } of circles) {
		ctx.beginPath();
		ctx.arc(x, y, CIRCLE_RADIUS, 0, Math.PI * 2);
		ctx.fill();
	}
	// Dessin du carré
	ctx.shadowBlur = 5;
	ctx.shadowColor = "white";
	ctx.fillStyle = '#dbdbdb';
	ctx.fillRect(squareX, squareY, SQUARE_SIZE, SQUARE_SIZE);


}


function visualTestGameLoop() {
	if (!isTestingVisual) return;
	const { stateFeatures } = getCurrentState();
	const actionIndex = bestAgentInstance.selectAction(stateFeatures);
	const { reward, collectedCount } = stepEnvironmentAndGetReward(actionIndex);

	currentTestScoreSpan.textContent = (parseFloat(currentTestScoreSpan.textContent) + reward).toFixed(2);
	circlesCollectedTestSpan.textContent = parseInt(circlesCollectedTestSpan.textContent) + collectedCount;

	drawGame();
	animationFrameId = requestAnimationFrame(visualTestGameLoop);
}

function startVisualTest() {
	if (!bestAgentInstance) return log("Aucune IA entraînée à tester.");
	if (isTraining) return log("Arrêtez l'entraînement avant de tester.");

	isTestingVisual = true;
	testBestIAButton.disabled = true;
	stopTestButtonVisual.disabled = false;
	startButton.disabled = true;

	resetGameEnvironment();
	currentTestScoreSpan.textContent = '0';
	circlesCollectedTestSpan.textContent = '0';
	log("Démarrage du test visuel...");
	visualTestGameLoop();
}

function stopVisualTest() {
	isTestingVisual = false;
	if (animationFrameId) {
		cancelAnimationFrame(animationFrameId);
		animationFrameId = null;
	}
	testBestIAButton.disabled = false;
	stopTestButtonVisual.disabled = true;
	startButton.disabled = false;
	drawGame();
	log("Test visuel arrêté.");
}

// --- GRAPHIQUE ET ÉVÉNEMENTS UI ---

function initLearningCurveChart() {
	const layout = {
		title: {
			text: '',
			font: {
				color: '#e0e0e0',
				family: 'Inter, sans-serif'
			}
		},
		xaxis: {
			title: 'Génération',
			color: '#b0b0b0',
			gridcolor: '#444444',
			linecolor: '#555555',
			tickfont: { color: '#b0b0b0' }
		},
		yaxis: {
			title: {
				text: 'Récompense Moyenne',
				standoff: 12
			},
			color: '#b0b0b0',
			automargin: true,
			gridcolor: '#444444',
			linecolor: '#555555',
			tickfont: { color: '#b0b0b0' }
		},
		plot_bgcolor: '#2a2a2a',
		paper_bgcolor: '#2a2a2a',
		margin: { l: 50, r: 25, b: 50, t: 50, pad: 4 }
	};
	const config = { responsive: true };

	Plotly.newPlot(learningCurveChartDiv, [{
		y: learningCurveData,
		mode: 'lines+markers',
		type: 'scatter',
		line: { color: '#9063ff', width: 3 },
		marker: { color: '#a07aff', size: 1 }
	}], layout, config);
}

function updateLearningCurveChart(newReward) {
	if (newReward === null) {
		// Cas spécial pour redessiner le graphique avec les données existantes (après chargement)
		Plotly.restyle(learningCurveChartDiv, { y: [learningCurveData] });
		return;
	}
	learningCurveData.push(newReward);
	Plotly.restyle(learningCurveChartDiv, { y: [learningCurveData] });
}

function resetLearningCurveChart() {
	learningCurveData = [];
	// Mettre à jour le graphique existant avec des données vides
	Plotly.restyle(learningCurveChartDiv, { y: [[]] });
}

function resetSettingsToDefault() {
	// Paramètres principaux
	numIndividualsInput.value = 10;
	numGenerationsInput.value = 100;
	infiniteGenerationsCheckbox.checked = false;
	episodesPerIndividualInput.value = 50;
	mutationRateInput.value = 0.05;
	learningRateActorInput.value = 0.001;
	learningRateCriticInput.value = 0.005;
	gammaInput.value = 0.99;
	numElitismInput.value = 1;

	// Paramètres de Récompense/Pénalité
	collectionDistanceThresholdInput.value = 10;
	stagnationPenaltyInput.value = -0.2;
	stagnationThresholdStepsInput.value = 30;
	actionChangePenaltyInput.value = -0.02;

	// S'assurer que les autres paramètres de l'UI (s'ils existent dans l'HTML) sont aussi réinitialisés
	// Exemple pour les paramètres bonus/malus qui étaient dans l'HTML original
	const bonusProximity = document.getElementById('bonusProximity');
	if (bonusProximity) bonusProximity.value = 0.2;
	const proximityThreshold = document.getElementById('proximityThreshold');
	if (proximityThreshold) proximityThreshold.value = 120;
	const bonusActionVariation = document.getElementById('bonusActionVariation');
	if (bonusActionVariation) bonusActionVariation.value = 0.005;
	const bonusNoWallHit = document.getElementById('bonusNoWallHit');
	if (bonusNoWallHit) bonusNoWallHit.value = 1;
	const penaltyPerStep = document.getElementById('penaltyPerStep');
	if (penaltyPerStep) penaltyPerStep.value = -0.01;
	const penaltyWallHit = document.getElementById('penaltyWallHit');
	if (penaltyWallHit) penaltyWallHit.value = -0.5;
	const penaltyRepetitiveAction = document.getElementById('penaltyRepetitiveAction');
	if (penaltyRepetitiveAction) penaltyRepetitiveAction.value = -0.03;
	const penaltyEarlyEnd = document.getElementById('penaltyEarlyEnd');
	if (penaltyEarlyEnd) penaltyEarlyEnd.value = -5;

	log("Tous les paramètres ont été réinitialisés à leurs valeurs par défaut.");
}

/**
 * MODIFICATION: Crée un objet de données prêt à être sauvegardé.
 * @returns {object}
 */
function prepareDataForSaving() {
	return {
		actorWeights: Array.from(bestAgentInstance.getActorWeights()),
		criticWeights: Array.from(bestAgentInstance.getCriticWeights()),
		bestReward: bestRewardAllTime,
		bestCircles: bestCirclesAllTime,
		currentGeneration: currentGeneration,
		learningCurve: learningCurveData,
		settings: {
			numIndividuals: numIndividualsInput.value,
			numGenerations: numGenerationsInput.value,
			infiniteGenerations: infiniteGenerationsCheckbox.checked,
			episodesPerIndividual: episodesPerIndividualInput.value,
			mutationRate: mutationRateInput.value,
			learningRateActor: learningRateActorInput.value,
			learningRateCritic: learningRateCriticInput.value,
			gamma: gammaInput.value,
			numElitism: numElitismInput.value,
		}
	};
}

/**
 * MODIFICATION: Applique les données chargées (d'un fichier ou d'IndexedDB) à l'état de l'application.
 * @param {object} loadedData - Les données à appliquer.
 */
function applyLoadedData(loadedData) {
	if (!loadedData || !loadedData.actorWeights || !loadedData.criticWeights) {
		log("Données de chargement invalides ou manquantes.");
		return;
	}

	if (isTraining) stopTraining();
	if (isTestingVisual) stopVisualTest();

	// Mettre à jour les paramètres de l'UI depuis les données
	if (loadedData.settings) {
		numIndividualsInput.value = loadedData.settings.numIndividuals;
		numGenerationsInput.value = loadedData.settings.numGenerations;
		infiniteGenerationsCheckbox.checked = loadedData.settings.infiniteGenerations;
		episodesPerIndividualInput.value = loadedData.settings.episodesPerIndividual;
		mutationRateInput.value = loadedData.settings.mutationRate;
		learningRateActorInput.value = loadedData.settings.learningRateActor;
		learningRateCriticInput.value = loadedData.settings.learningRateCritic;
		gammaInput.value = loadedData.settings.gamma;
		numElitismInput.value = loadedData.settings.numElitism;
	}

	if (stateFeatureSize === 0) {
		resetGameEnvironment();
		stateFeatureSize = getCurrentState().stateFeatures.length;
	}

	bestAgentInstance = new ActorCritic(
		parseFloat(learningRateActorInput.value),
		parseFloat(learningRateCriticInput.value),
		parseFloat(gammaInput.value),
		stateFeatureSize,
		8, // numActions
		loadedData.actorWeights,
		loadedData.criticWeights
	);

	bestRewardAllTime = loadedData.bestReward || 0;
	bestCirclesAllTime = loadedData.bestCircles || 0;
	currentGeneration = loadedData.currentGeneration || 0;
	learningCurveData = loadedData.learningCurve || [];

	bestRewardAllTimeSpan.textContent = bestRewardAllTime.toFixed(2);
	bestCirclesAllTimeSpan.textContent = bestCirclesAllTime;
	currentGenerationSpan.textContent = currentGeneration;
	updateLearningCurveChart(null); // Redessine le graphique

	testBestIAButton.disabled = false;
	log("Agent et statistiques chargés avec succès.");
}


// --- ÉCOUTEURS D'ÉVÉNEMENTS ---

resetButton.addEventListener('click', async () => {
	const confirm = await showCustomConfirm("Voulez-vous vraiment tout réinitialiser ? Cela supprimera la sauvegarde automatique.");
	if (confirm) {
		if (isTraining) stopTraining();
		if (isTestingVisual) stopVisualTest();

		bestAgentInstance = null;
		bestRewardAllTime = 0;
		bestCirclesAllTime = 0;
		currentGeneration = 0;

		bestRewardAllTimeSpan.textContent = '0';
		bestCirclesAllTimeSpan.textContent = '0';
		currentGenerationSpan.textContent = '0';

		resetLearningCurveChart();
		resetSettingsToDefault();

		// MODIFICATION : Vide la base de données IndexedDB
		await clearIndexedDB();

		resetGameEnvironment();
		drawGame();
		log("Réinitialisation complète effectuée.");
	}
});

startButton.addEventListener('click', runGenerations);
stopButton.addEventListener('click', () => stopTraining(true));
testBestIAButton.addEventListener('click', startVisualTest);
stopTestButtonVisual.addEventListener('click', stopVisualTest);
settingsButton.addEventListener('click', () => settingsPanel.classList.toggle('open'));
closeButton.addEventListener('click', () => settingsPanel.classList.toggle('open'));
saveButton.addEventListener('click', () => {
	if (!bestAgentInstance) {
		return log("Aucun agent entraîné à sauvegarder.");
	}
	const dataToSave = prepareDataForSaving();
	const jsonString = JSON.stringify(dataToSave, null, 2);
	const blob = new Blob([jsonString], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = `agent_gen_${currentGeneration}.json`;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
	log("Agent sauvegardé dans un fichier JSON.");
});

loadButton.addEventListener('click', () => {
	loadFileInput.click();
});

loadFileInput.addEventListener('change', (event) => {
	const file = event.target.files[0];
	if (!file) return;

	const reader = new FileReader();
	reader.onload = (e) => {
		try {
			const loadedData = JSON.parse(e.target.result);
			// MODIFICATION : Utilise la nouvelle fonction centralisée
			applyLoadedData(loadedData);
		} catch (err) {
			console.error("Erreur lors du chargement du fichier :", err);
			log("Échec du chargement du fichier.");
		}
	};
	reader.readAsText(file);
	event.target.value = '';
});



// --- INITIALISATION ---
window.onload = async function () {
	initializeWorkers();
	resetGameEnvironment();
	drawGame();
	initLearningCurveChart();

	// MODIFICATION : Chargement automatique depuis IndexedDB
	const savedData = await loadAgentFromDB();
	if (savedData) {
		log("Agent précédemment sauvegardé trouvé. Chargement...");
		applyLoadedData(savedData);
	} else {
		log("Aucune sauvegarde automatique trouvée.");
	}
	if (loaderOverlay) {
		loaderOverlay.style.display = 'none';
	}
	log("Application prête.");
};