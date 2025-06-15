// ia.js - Implémentation de l'IA Actor-Critic (avec support génétique)
// et logique principale du jeu

// Import de la classe ActorCritic depuis son propre fichier
import { ActorCritic } from './actorCritic.js';

// Déclaration des variables globales pour le DOM, le jeu et l'entraînement
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const numIndividualsInput = document.getElementById('numIndividuals');
const numGenerationsInput = document.getElementById('numGenerations');
const infiniteGenerationsCheckbox = document.getElementById('infiniteGenerationsCheckbox'); // Nouvelle référence
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

// Références pour le panneau de paramètres
const settingsPanel = document.getElementById('settingsPanel');
const settingsButton = document.getElementById('settingsButton');

// Références pour les nouveaux boutons de sauvegarde/chargement
const saveButton = document.getElementById('saveButton');
const loadButton = document.getElementById('loadButton');
const loadFileInput = document.getElementById('loadFileInput'); // Input de fichier caché
const resetButton = document.getElementById('resetButton');

// Références pour la modale de confirmation
const customConfirmModal = document.getElementById('customConfirmModal');
const modalMessage = document.getElementById('modalMessage');
const modalConfirmBtn = document.getElementById('modalConfirmBtn');
const modalCancelBtn = document.getElementById('modalCancelBtn');

// L'élément logOutput n'est plus nécessaire car le journal est supprimé de l'UI
// const logOutput = document.getElementById('logOutput');

// --- Variables pour le graphique de la courbe d'apprentissage ---
const learningCurveChartDiv = document.getElementById('learningCurveChart');
let learningCurveData = []; // Stocke les récompenses moyennes par génération

// --- Paramètres du Jeu (pour l'environnement d'évaluation) ---
const SQUARE_SIZE = 30;
let squareX = canvas.width / 2 - SQUARE_SIZE / 2;
let squareY = canvas.height / 2 - SQUARE_SIZE / 2;
const SQUARE_SPEED = 5;

const CIRCLE_RADIUS = 10;
const MAX_CIRCLES = 5;
let circles = [];

// Statistiques globales de la session (non réinitialisées à chaque génération)
let bestRewardAllTime = 0;
let bestCirclesAllTime = 0;

let isTraining = false;
let isTestingVisual = false;
let animationFrameId = null;
let currentGeneration = 0; // Variable globale pour suivre le nombre réel de générations

// Stocke les poids du meilleur agent pour le test
let bestActorWeights = null;
let bestCriticWeights = null;
let bestAgentInstance = null;

// Indique si le rendu du canvas est activé (pour le test visuel)
let isCanvasRenderingEnabled = true;

// --- Paramètres de Récompense / Pénalité ---
// Valeurs par défaut, seront mises à jour depuis l'UI
let REWARD_COLLECT_CIRCLE = 10;
let PENALTY_PER_STEP = -0.01;

let BONUS_QUICK_COLLECTION_BASE = 5;
let BONUS_QUICK_COLLECTION_DECAY = 0.02;

let BONUS_PROXIMITY = 0.05;
let PROXIMITY_THRESHOLD = 150;

let BONUS_ACTION_VARIATION_PER_STEP = 0.005;
let ACTION_HISTORY_BUFFER_SIZE = 10;
let MIN_UNIQUE_ACTIONS_FOR_BONUS = 3;

let BONUS_NO_WALL_HIT_EPISODE = 1;

let PENALTY_WALL_HIT_PER_STEP = -0.5;
let PENALTY_INACTIVITY = -0.1;
let INACTIVITY_THRESHOLD_STEPS = 50;

let PENALTY_REPETITIVE_ACTION = -0.05;
let REPETITIVE_ACTION_THRESHOLD_STEPS = 10;

let PENALTY_EARLY_EMPTY_EPISODE = -5;
let EARLY_END_TIME_THRESHOLD = 100;

let COLLECTION_DISTANCE_THRESHOLD = 10;
let PENALTY_STAGNATION = -0.2;
let STAGNATION_THRESHOLD_STEPS = 30;
let PENALTY_ACTION_CHANGE = -0.02;

// Variables d'état pour le calcul de récompense par épisode
let lastSquareX;
let lastSquareY;
let inactivitySteps;
let lastActionIndex = -1;
let repetitiveActionSteps;
let episodeWallHit;
let actionHistoryBuffer = [];
let previousDistanceToClosestCircle;
let stagnationTrackingSteps = 0;


// Déclaration de la variable globale pour la taille des états
let stateFeatureSize = 0;


// --- IndexedDB Setup ---
const DB_NAME = 'ActorCriticTrainingDB';
const STORE_NAME = 'agentWeights';
const DB_VERSION = 1;
let db;

/**
 * Ouvre ou crée la base de données IndexedDB.
 * @returns {Promise<IDBDatabase>} La base de données IndexedDB.
 */
function openDatabase() {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, DB_VERSION);

		request.onupgradeneeded = (event) => {
			db = event.target.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME, { keyPath: 'id' });
			}
		};

		request.onsuccess = (event) => {
			db = event.target.result;
			log("IndexedDB ouvert avec succès.");
			resolve(db);
		};

		request.onerror = (event) => {
			console.error("Erreur IndexedDB:", event.target.errorCode);
			reject(event.target.errorCode);
		};
	});
}

/**
 * Sauvegarde les poids de l'acteur et du critique dans IndexedDB.
 * @param {Array<Array<number>>} actorWeights - Poids de l'acteur.
 * @param {Array<number>} criticWeights - Poids du critique.
 * @param {number} reward - Meilleure récompense globale.
 * @param {number} circlesCollected - Nombre de ronds collectés pour la meilleure récompense.
 * @param {number} currentGen - La génération actuelle.
 * @param {Array<number>} learningCurveData - Données de la courbe d'apprentissage.
 * @param {boolean} infiniteMode - État du mode infini.
 */
async function saveWeightsToIndexedDB(actorWeights, criticWeights, reward, circlesCollected, currentGen, learningCurveData, infiniteMode) {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("Échec de l'ouverture d'IndexedDB pour la sauvegarde:", error);
			return;
		}
	}

	const transaction = db.transaction([STORE_NAME], 'readwrite');
	const store = transaction.objectStore(STORE_NAME);

	const data = {
		id: 'bestAgent', // Clé unique pour stocker un seul meilleur agent
		actorWeights: actorWeights,
		criticWeights: criticWeights,
		bestReward: reward,
		bestCircles: circlesCollected,
		currentGeneration: currentGen, // Sauvegarde la génération actuelle
		learningCurve: learningCurveData, // Sauvegarde les données du graphique
		infiniteGenerationsActive: infiniteMode, // Sauvegarde l'état du mode infini
		timestamp: new Date().toISOString()
	};

	const request = store.put(data);

	request.onsuccess = () => {
		log("Poids de l'agent sauvegardés dans IndexedDB.");
	};

	request.onerror = (event) => {
		console.error("Erreur lors de la sauvegarde dans IndexedDB:", event.target.error);
	};
}

/**
 * Charge les poids de l'acteur et du critique depuis IndexedDB.
 * @returns {Promise<object|null>} L'objet contenant les poids et autres données ou null si non trouvé.
 */
async function loadWeightsFromIndexedDB() {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("Échec de l'ouverture d'IndexedDB pour le chargement:", error);
			return null;
		}
	}

	return new Promise((resolve) => {
		const transaction = db.transaction([STORE_NAME], 'readonly');
		const store = transaction.objectStore(STORE_NAME);
		const request = store.get('bestAgent');

		request.onsuccess = (event) => {
			const data = event.target.result;
			if (data) {
				log("Poids de l'agent chargés depuis IndexedDB.");
				resolve({
					actorWeights: data.actorWeights,
					criticWeights: data.criticWeights,
					bestReward: data.bestReward,
					bestCircles: data.bestCircles,
					currentGeneration: data.currentGeneration || 0, // Charge la génération, ou 0 par défaut
					learningCurve: data.learningCurve || [], // Charge les données de la courbe, ou [] par default
					infiniteGenerationsActive: data.infiniteGenerationsActive || false // Charge l'état du mode infini
				});
			} else {
				log("Aucun poids d'agent trouvé dans IndexedDB.");
				resolve(null);
			}
		};

		request.onerror = (event) => {
			console.error("Erreur lors du chargement depuis IndexedDB:", event.target.error);
			resolve(null);
		};
	});
}

/**
 * Efface toutes les données de la base de données IndexedDB.
 */
async function clearIndexedDB() {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("Échec de l'ouverture d'IndexedDB pour l'effacement:", error);
			return;
		}
	}

	return new Promise((resolve) => {
		const transaction = db.transaction([STORE_NAME], 'readwrite');
		const store = transaction.objectStore(STORE_NAME);
		const request = store.clear();

		request.onsuccess = () => {
			log("IndexedDB effacé avec succès.");
		};

		request.onerror = (event) => {
			console.error("Erreur lors de l'effacement d'IndexedDB:", event.target.error);
		};
	});
}


// --- Fonctions utilitaires ---
function getRandomInt(min, max) {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Enregistre un message dans la console du navigateur.
 * Cette fonction ne met plus à jour l'élément DOM 'logOutput'.
 * @param {string} message - Le message à enregistrer.
 */
function log(message) {
	// Utilise un niveau de verbosité ou un drapeau debug si nécessaire pour limiter les logs
	// if (DEBUG_MODE) {
	console.log(`[${new Date().toLocaleTimeString()}] ${message}`);
	// }
}

/**
 * Affiche une modale de confirmation personnalisée.
 * @param {string} message - Le message à afficher dans la modale.
 * @returns {Promise<boolean>} Résout à true si l'utilisateur confirme, false sinon.
 */
function showCustomConfirm(message) {
	return new Promise((resolve) => {
		modalMessage.textContent = message;
		customConfirmModal.style.display = 'flex'; // Afficher la modale

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


// Fonction pour mettre à jour les paramètres de récompense/pénalité depuis l'UI
function updateRewardParameters() {
	COLLECTION_DISTANCE_THRESHOLD = parseFloat(collectionDistanceThresholdInput.value);
	PENALTY_STAGNATION = parseFloat(stagnationPenaltyInput.value);
	STAGNATION_THRESHOLD_STEPS = parseInt(stagnationThresholdStepsInput.value);
	PENALTY_ACTION_CHANGE = parseFloat(actionChangePenaltyInput.value);
	BONUS_PROXIMITY = parseFloat(document.getElementById('bonusProximity').value);
	PROXIMITY_THRESHOLD = parseFloat(document.getElementById('proximityThreshold').value);
	BONUS_ACTION_VARIATION_PER_STEP = parseFloat(document.getElementById('bonusActionVariation').value);
	BONUS_NO_WALL_HIT_EPISODE = parseFloat(document.getElementById('bonusNoWallHit').value);

	PENALTY_PER_STEP = parseFloat(document.getElementById('penaltyPerStep').value);
	PENALTY_WALL_HIT_PER_STEP = parseFloat(document.getElementById('penaltyWallHit').value);
	PENALTY_REPETITIVE_ACTION = parseFloat(document.getElementById('penaltyRepetitiveAction').value);
	PENALTY_EARLY_EMPTY_EPISODE = parseFloat(document.getElementById('penaltyEarlyEnd').value);
}


// --- Fonctions de l'Environnement (pour l'évaluation des individus) ---
function generateCircle() {
	let newCircleX, newCircleY;
	let collision;
	do {
		collision = false;
		newCircleX = getRandomInt(CIRCLE_RADIUS, canvas.width - CIRCLE_RADIUS);
		newCircleY = getRandomInt(CIRCLE_RADIUS, canvas.height - CIRCLE_RADIUS);
		// Vérifier la collision avec le carré
		if (newCircleX + SQUARE_SIZE > squareX && newCircleX - CIRCLE_RADIUS < squareX + SQUARE_SIZE &&
			newCircleY + CIRCLE_RADIUS > squareY && newCircleY - CIRCLE_RADIUS < squareY + SQUARE_SIZE) {
			collision = true;
		}
		// Vérifier la collision avec les cercles existants
		for (const existingCircle of circles) {
			const dx = newCircleX - existingCircle.x;
			const dy = newCircleY - existingCircle.y;
			const distanceSquared = dx * dx + dy * dy; // Optimisation: comparer les carrés
			if (distanceSquared < (CIRCLE_RADIUS * 2) * (CIRCLE_RADIUS * 2)) { // S'assurer que les cercles ne se chevauchent pas
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

	// Réinitialisation des variables d'état pour les récompenses
	lastSquareX = squareX;
	lastSquareY = squareY;
	inactivitySteps = 0;
	lastActionIndex = -1;
	repetitiveActionSteps = 0;
	episodeWallHit = false;
	actionHistoryBuffer = [];
	previousDistanceToClosestCircle = undefined;
	stagnationTrackingSteps = 0;
}

// Fonction utilitaire pour calculer l'état de l'environnement
// Ne doit être appelée qu'une seule fois par étape pour obtenir l'état courant
function getCurrentState() {
	let closestCircle = null;
	let minDistanceSquared = Infinity; // Optimisation: travailler avec les distances au carré

	if (circles.length === 0) {
		generateCircle();
	}

	circles.forEach(circle => {
		const dx = squareX + SQUARE_SIZE / 2 - circle.x;
		const dy = squareY + SQUARE_SIZE / 2 - circle.y;
		const distanceSquared = dx * dx + dy * dy; // Calculer la distance au carré une seule fois
		if (distanceSquared < minDistanceSquared) {
			minDistanceSquared = distanceSquared;
			closestCircle = circle;
		}
	});

	const minDistance = Math.sqrt(minDistanceSquared); // Prendre la racine carrée seulement si la distance réelle est nécessaire

	// Normalisation des valeurs de l'état entre 0 et 1
	const stateFeatures = [
		squareX / canvas.width,
		squareY / canvas.height,
		(closestCircle ? closestCircle.x : 0) / canvas.width,
		(closestCircle ? closestCircle.y : 0) / canvas.height,
		(closestCircle ? minDistance : 0) / Math.sqrt(canvas.width * canvas.width + canvas.height * canvas.height)
	];
	return { stateFeatures, minDistance, closestCircle, minDistanceSquared };
}

/**
 * Effectue un pas dans l'environnement, met à jour la position du carré,
 * vérifie les collisions et calcule la récompense pour ce pas.
 * @param {number} actionIndex - L'action choisie par l'agent (0:Haut, 1:Bas, 2:Gauche, 3:Droite, 4:Haut-Gauche, 5:Haut-Droite, 6:Bas-Gauche, 7:Bas-Droite).
 * @param {number} timeStep - Le numéro du pas de temps actuel dans l'épisode.
 * @param {object} currentStateData - Les données de l'état actuel (stateFeatures, minDistance, minDistanceSquared, closestCircle) obtenues de getCurrentState().
 * @returns {object} Un objet contenant la récompense pour ce pas et le nombre de ronds collectés.
 */
function stepEnvironmentAndGetReward(actionIndex, timeStep, currentStateData) {
	let currentReward = PENALTY_PER_STEP;
	let collectedCount = 0;

	// Utilise les données de l'état déjà calculées
	const { minDistance: currentDistanceToClosest, minDistanceSquared: currentDistanceSquaredToClosest, closestCircle } = currentStateData;

	if (previousDistanceToClosestCircle === undefined) {
		previousDistanceToClosestCircle = currentDistanceToClosest;
	}

	// Malus d'oscillation (Malus Supplémentaire)
	if (lastActionIndex !== -1 && actionIndex !== lastActionIndex) {
		currentReward += PENALTY_ACTION_CHANGE;
	}

	let nextSquareX = squareX;
	let nextSquareY = squareY;

	// Logique de déplacement pour 8 directions
	switch (actionIndex) {
		case 0: nextSquareY -= SQUARE_SPEED; break; // Haut
		case 1: nextSquareY += SQUARE_SPEED; break; // Bas
		case 2: nextSquareX -= SQUARE_SPEED; break; // Gauche
		case 3: nextSquareX += SQUARE_SPEED; break; // Droite
		case 4: nextSquareY -= SQUARE_SPEED; nextSquareX -= SQUARE_SPEED; break; // Haut-Gauche
		case 5: nextSquareY -= SQUARE_SPEED; nextSquareX += SQUARE_SPEED; break; // Haut-Droite
		case 6: nextSquareY += SQUARE_SPEED; nextSquareX -= SQUARE_SPEED; break; // Bas-Gauche
		case 7: nextSquareY += SQUARE_SPEED; nextSquareX += SQUARE_SPEED; break; // Bas-Droite
	}

	const willHitWall = (nextSquareX < 0 || nextSquareX > canvas.width - SQUARE_SIZE ||
		nextSquareY < 0 || nextSquareY > canvas.height - SQUARE_SIZE);
	if (willHitWall) {
		currentReward += PENALTY_WALL_HIT_PER_STEP;
		episodeWallHit = true;
	}

	squareX = Math.max(0, Math.min(canvas.width - SQUARE_SIZE, nextSquareX));
	squareY = Math.max(0, Math.min(canvas.height - SQUARE_SIZE, nextSquareY));

	// Détection d'inactivité (Malus 3)
	if (squareX === lastSquareX && squareY === lastSquareY) {
		inactivitySteps++;
	} else {
		inactivitySteps = 0;
	}
	lastSquareX = squareX;
	lastSquareY = squareY;

	// Malus de stagnation (Nouveau Malus)
	// Si immobile ET proche d'un cercle sans le ramasser, incrémente le compteur de stagnation
	if (inactivitySteps > 0 && currentDistanceToClosest > (CIRCLE_RADIUS + SQUARE_SIZE / 2) && currentDistanceToClosest < PROXIMITY_THRESHOLD) {
		stagnationTrackingSteps++;
		if (stagnationTrackingSteps >= STAGNATION_THRESHOLD_STEPS) {
			currentReward += PENALTY_STAGNATION;
		}
	} else {
		stagnationTrackingSteps = 0; // Réinitialise si bouge ou ramasse ou s'éloigne
	}


	// Détection d'actions répétitives (Malus 4) - seulement si pas de ronds ramassés
	const previousCollectedCount = collectedCount; // Sauvegarde l'état de collecte avant le check
	if (actionIndex === lastActionIndex && previousCollectedCount === 0) {
		repetitiveActionSteps++;
		if (repetitiveActionSteps >= REPETITIVE_ACTION_THRESHOLD_STEPS) {
			currentReward += PENALTY_REPETITIVE_ACTION;
		}
	} else {
		repetitiveActionSteps = 0;
	}
	lastActionIndex = actionIndex; // Met à jour pour le prochain pas

	// Mettre à jour l'historique des actions (Bonus 3: Variation d'actions)
	actionHistoryBuffer.push(actionIndex);
	if (actionHistoryBuffer.length > ACTION_HISTORY_BUFFER_SIZE) {
		actionHistoryBuffer.shift();
	}
	const uniqueActions = new Set(actionHistoryBuffer).size;
	if (uniqueActions >= MIN_UNIQUE_ACTIONS_FOR_BONUS) {
		currentReward += BONUS_ACTION_VARIATION_PER_STEP;
	}

	// Vérifie les collisions avec les cercles (Bonus 1 + Détection par distance)
	// On ne recalcule pas la distance, on itère sur tous les cercles
	for (let i = circles.length - 1; i >= 0; i--) {
		const circle = circles[i];
		const dx = (squareX + SQUARE_SIZE / 2) - circle.x;
		const dy = (squareY + SQUARE_SIZE / 2) - circle.y;
		const distanceSquared = dx * dx + dy * dy; // Calculer la distance au carré

		// Collection par distance (Nouveau comportement)
		// Comparaison directe des carrés pour éviter Math.sqrt inutilement
		if (distanceSquared < ((CIRCLE_RADIUS + SQUARE_SIZE / 2 + COLLECTION_DISTANCE_THRESHOLD) * (CIRCLE_RADIUS + SQUARE_SIZE / 2 + COLLECTION_DISTANCE_THRESHOLD))) {
			circles.splice(i, 1);
			currentReward += REWARD_COLLECT_CIRCLE;
			collectedCount++;

			// Bonus 2: Rapidité de collecte après le début de l'épisode
			currentReward += Math.max(0, BONUS_QUICK_COLLECTION_BASE - (timeStep * BONUS_QUICK_COLLECTION_DECAY));

			if (circles.length < MAX_CIRCLES) {
				generateCircle();
			}
		}
	}

	// Bonus 4: Se rapprocher d'un objet
	// Appeler getCurrentState() une seule fois pour obtenir nextState
	const { minDistance: newDistanceToClosest } = getCurrentState(); // Cet appel est inévitable pour le 'nextState'

	if (newDistanceToClosest < previousDistanceToClosestCircle && newDistanceToClosest > 0 && newDistanceToClosest < PROXIMITY_THRESHOLD) {
		currentReward += BONUS_PROXIMITY;
	}
	previousDistanceToClosestCircle = newDistanceToClosest;

	// Récupère le nextStateFeatures pour le training de l'Actor-Critic
	const { stateFeatures: nextStateFeatures } = getCurrentState();

	return { reward: currentReward, collectedCount: collectedCount, nextStateFeatures: nextStateFeatures };
}


/**
 * Évalue la performance d'un individu (agent ActorCritic) sur un nombre donné d'épisodes.
 * L'individu apprend (via Actor-Critic) pendant ces épisodes si isTrainingRun est vrai.
 * @param {ActorCritic} agent - L'instance de l'agent à évaluer.
 * @param {number} numEpisodes - Le nombre d'épisodes pour l'évaluation.
 * @param {boolean} isTrainingRun - Vrai si c'est un run d'entraînement (avec apprentissage), faux sinon.
 * @returns {object} Un objet contenant la récompense totale accumulée et les ronds ramassés.
 */
async function evaluateIndividual(agent, numEpisodes, isTrainingRun = true) {
	let totalReward = 0;
	let totalCirclesCollected = 0;
	const MAX_TIME_STEPS_PER_EPISODE = 300;

	for (let i = 0; i < numEpisodes; i++) {
		if (!isTraining) {
			log("evaluateIndividual: Entraînement interrompu (début épisode).");
			return { totalReward, totalCirclesCollected };
		}

		resetGameEnvironment();
		let episodeReward = 0;
		let episodeCirclesCollected = 0;
		let timeStep = 0;
		let episodeTerminatedEarlyWithoutCollection = false;

		// Optimisation: Obtenir l'état initial une seule fois
		let { stateFeatures: currentState, minDistance: initialMinDistance, minDistanceSquared: initialMinDistanceSquared, closestCircle: initialClosestCircle } = getCurrentState();

		while (timeStep < MAX_TIME_STEPS_PER_EPISODE) {
			if (!isTraining) {
				log("evaluateIndividual: Entraînement interrompu (dans boucle pas de temps).");
				break;
			}

			timeStep++;
			const actionIndex = agent.selectAction(currentState);

			// Passer l'état actuel complet pour éviter les recalculs internes
			const { reward: stepReward, collectedCount: stepCollectedCount, nextStateFeatures: nextState } = stepEnvironmentAndGetReward(actionIndex, timeStep, {
				stateFeatures: currentState,
				minDistance: initialMinDistance, // Utiliser le minDistance initial ou le mettre à jour si nécessaire
				minDistanceSquared: initialMinDistanceSquared,
				closestCircle: initialClosestCircle
			});


			episodeReward += stepReward;
			episodeCirclesCollected += stepCollectedCount;

			if (isTrainingRun) {
				const done = (timeStep >= MAX_TIME_STEPS_PER_EPISODE || circles.length === 0);
				agent.train(currentState, actionIndex, stepReward, nextState, done);
			}

			// Mettre à jour l'état pour le pas suivant
			currentState = nextState;

			if (circles.length === 0 || timeStep >= MAX_TIME_STEPS_PER_EPISODE) {
				episodeTerminatedEarlyWithoutCollection = (episodeCirclesCollected === 0 && timeStep < EARLY_END_TIME_THRESHOLD);
				break;
			}
		}
		totalReward += episodeReward;
		totalCirclesCollected += episodeCirclesCollected;

		// Bonus 5: Si l'agent n'a pas heurté de mur pendant tout l'épisode
		if (!episodeWallHit) {
			totalReward += BONUS_NO_WALL_HIT_EPISODE;
		}

		// Malus 5: Si l'épisode se termine trop vite sans rien ramasser
		if (episodeTerminatedEarlyWithoutCollection) {
			totalReward += PENALTY_EARLY_EMPTY_EPISODE;
		}
	}
	return { totalReward, totalCirclesCollected };
}

/**
 * Croise les poids de deux parents pour créer un nouvel enfant (uniform crossover).
 * @param {Array<Array<number>>} parent1Weights - Les poids du premier parent.
 * @param {Array<Array<number>>} parent2Weights - Les poids du second parent.
 * @returns {Array<Array<number>>} Les poids du nouvel enfant.
 */
function crossover(parent1Weights, parent2Weights) {
	const childWeights = JSON.parse(JSON.stringify(parent1Weights));
	for (let i = 0; i < parent1Weights.length; i++) {
		for (let j = 0; j < parent1Weights[i].length; j++) {
			if (Math.random() < 0.5) {
				childWeights[i][j] = parent2Weights[i][j];
			}
		}
	}
	return childWeights;
}

/**
 * Applique une mutation aux poids d'un individu.
 * @param {Array<Array<number>>} weights - Les poids à muter.
 * @param {number} mutationRate - La probabilité qu'un poids mutate.
 * @param {number} mutationStrength - L'ampleur de la mutation.
 * @returns {Array<Array<number>>} Les poids mutés.
 */
function mutate(weights, mutationRate, mutationStrength = 0.1) {
	for (let i = 0; i < weights.length; i++) {
		for (let j = 0; j < weights[i].length; j++) {
			if (Math.random() < mutationRate) {
				weights[i][j] += (Math.random() * 2 - 1) * mutationStrength;
			}
		}
	}
	return weights;
}

// --- Fonctions Plotly pour le graphique de la courbe d'apprentissage ---

/**
 * Initialise le graphique de la courbe d'apprentissage avec Plotly.
 */
function initLearningCurveChart() {
	const layout = {
		title: {
			text: 'Récompense Moyenne par Génération',
			font: {
				color: '#e0e0e0', // Couleur du texte du titre
				family: 'Inter, sans-serif'
			}
		},
		xaxis: {
			title: 'Génération',
			color: '#b0b0b0', // Couleur du texte de l'axe X
			gridcolor: '#444444', // Couleur des lignes de grille
			linecolor: '#555555', // Couleur de la ligne de l'axe
			tickfont: {
				color: '#b0b0b0' // Couleur des graduations
			}
		},
		yaxis: {
			title: 'Récompense Moyenne',
			color: '#b0b0b0', // Couleur du texte de l'axe Y
			gridcolor: '#444444',
			linecolor: '#555555',
			tickfont: {
				color: '#b0b0b0'
			}
		},
		plot_bgcolor: '#1a1a1a', // Couleur de fond du graphique (zone de tracé)
		paper_bgcolor: '#1a1a1a', // Couleur de fond de l'ensemble du papier Plotly
		margin: {
			l: 50,
			r: 50,
			b: 50,
			t: 50
		},
		hovermode: 'closest' // Afficher les info-bulles pour le point le plus proche
	};

	const config = {
		responsive: true // Le graphique sera réactif
	};

	Plotly.newPlot(learningCurveChartDiv, [{
		y: learningCurveData,
		mode: 'lines+markers',
		type: 'scatter',
		name: 'Récompense Moyenne',
		line: {
			color: '#00e676', // Couleur de la ligne du graphique (accent-color-green)
			width: 2
		},
		marker: {
			color: '#ffeb3b', // Couleur des marqueurs (accent-color-yellow)
			size: 6,
			symbol: 'circle'
		}
	}], layout, config);
}

/**
 * Met à jour le graphique de la courbe d'apprentissage avec de nouvelles données.
 * @param {number} newReward - La récompense moyenne de la génération actuelle.
 */
function updateLearningCurveChart(newReward) {
	learningCurveData.push(newReward);

	Plotly.update(learningCurveChartDiv, {
		y: [learningCurveData] // Mettre à jour l'axe Y du premier trace (index 0)
	});
}

/**
 * Réinitialise le graphique de la courbe d'apprentissage.
 */
function resetLearningCurveChart() {
	learningCurveData = []; // Efface toutes les données
	// Recréer le graphique vide pour s'assurer que toutes les propriétés sont réinitialisées
	initLearningCurveChart();
	log("Graphique de la courbe d'apprentissage réinitialisé.");
}


// --- Fonctions de gestion des Web Workers ---
// Limiter le nombre de workers à une valeur plus conservative (par exemple, 4 ou 8)
// pour éviter la surcharge du CPU sur les machines ayant de nombreux cœurs logiques.
const NUM_WORKERS = Math.min(12, Math.max(2, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 2));
const workers = [];
// workerPromises sera géré par evaluateWithWorkerPool

/**
 * Initialise les Web Workers.
 */
function initializeWorkers() {
	// S'assurer que les workers ne sont pas initialisés plusieurs fois
	if (workers.length === 0) {
		for (let i = 0; i < NUM_WORKERS; i++) {
			const worker = new Worker('./src/worker.js', { type: 'module' }); // Assurez-vous que worker.js est au bon endroit
			workers.push(worker);
		}
		log(`${NUM_WORKERS} Web Workers initialisés.`);
	} else {
		log(`Web Workers déjà initialisés (${NUM_WORKERS} workers).`);
	}
}

/**
 * Envoie une tâche à un Web Worker et retourne une promesse qui se résout avec le résultat.
 * Utilise addEventListener/removeEventListener pour une gestion propre.
 * @param {Worker} worker - L'instance du Web Worker.
 * @param {object} data - Les données à envoyer au worker.
 * @returns {Promise<object>} La promesse qui résout avec les résultats du worker.
 */
function runWorkerTask(worker, data) {
	return new Promise((resolve, reject) => {
		const onMessage = (e) => {
			resolve(e.data);
			worker.removeEventListener('message', onMessage); // Nettoyer l'écouteur après utilisation
			worker.removeEventListener('error', onError); // S'assurer que l'écouteur d'erreur est aussi nettoyé
		};
		const onError = (e) => {
			console.error(`Worker error: ${e.message} at ${e.filename}:${e.lineno}`);
			reject(e);
			worker.removeEventListener('message', onMessage); // Nettoyer l'écouteur après erreur
			worker.removeEventListener('error', onError); // S'assurer que l'écouteur d'erreur est aussi nettoyé
		};

		worker.addEventListener('message', onMessage);
		worker.addEventListener('error', onError);
		worker.postMessage(data);
	});
}

/**
 * Gère l'exécution des tâches d'évaluation des individus en utilisant un pool de Web Workers.
 * @param {Array<object>} tasks - Un tableau d'objets de tâches, chaque objet contenant agentBrain et simulationParams.
 * @returns {Promise<Array<object>>} Un tableau des résultats (fulfilled ou rejected) pour chaque tâche.
 */
async function evaluateWithWorkerPool(tasks) {
	const results = Array(tasks.length).fill(null); // Pré-allouer le tableau pour les résultats
	let currentTaskIndex = 0;
	let runningWorkers = 0;

	return new Promise((resolve) => {
		const processNextTask = () => {
			// Si toutes les tâches ont été soumises et tous les workers sont inactifs, résoudre la promesse
			if (currentTaskIndex >= tasks.length && runningWorkers === 0) {
				resolve(results);
				return;
			}

			// Tant qu'il y a des workers disponibles et des tâches en attente
			while (runningWorkers < NUM_WORKERS && currentTaskIndex < tasks.length) {
				const taskToRunIndex = currentTaskIndex++;
				const workerToUse = workers[runningWorkers % NUM_WORKERS]; // Simple répartition circulaire
				runningWorkers++; // Incrémenter le compteur de workers occupés

				// Exécuter la tâche sur le worker
				runWorkerTask(workerToUse, tasks[taskToRunIndex])
					.then((result) => {
						results[taskToRunIndex] = { status: 'fulfilled', value: result };
					})
					.catch((error) => {
						results[taskToRunIndex] = { status: 'rejected', reason: error };
						console.error(`Erreur dans la tâche ${taskToRunIndex}:`, error);
					})
					.finally(() => {
						runningWorkers--; // Décrémenter le compteur de workers occupés
						processNextTask(); // Tenter de lancer la prochaine tâche
					});
			}
		};

		// Lancer le traitement des premières tâches
		processNextTask();
	});
}


/**
 * Fonction principale de l'algorithme génétique.
 * @param {ActorCritic|null} initialBestAgent - L'agent le mieux entraîné à partir duquel reprendre.
 */
async function runGenerations(initialBestAgent = null) {
	log("Démarrage de l'entraînement génétique...");
	testBestIAButton.disabled = true;
	stopTestButtonVisual.disabled = true;
	isCanvasRenderingEnabled = false; // Désactiver le rendu du canvas pendant l'entraînement

	const numIndividuals = parseInt(numIndividualsInput.value);
	const numGenerations = parseInt(numGenerationsInput.value);
	const episodesPerIndividual = parseInt(episodesPerIndividualInput.value);
	const mutationRate = parseFloat(mutationRateInput.value);
	const learningRateActor = parseFloat(learningRateActorInput.value);
	const learningRateCritic = parseFloat(learningRateCriticInput.value);
	const gamma = parseFloat(gammaInput.value);
	const numElitism = parseInt(numElitismInput.value);
	const isInfiniteMode = infiniteGenerationsCheckbox.checked; // Récupère l'état du mode infini

	// Mettre à jour les paramètres de récompense/pénalité au début de chaque run d'entraînement
	updateRewardParameters();

	let population = [];
	const NUM_ACTIONS = 8; // Le nombre total d'actions

	// Optimisation: Ne pas appeler getCurrentState() dans la boucle pour la taille des états
	if (stateFeatureSize === 0) { // Si non initialisé, le faire une fois
		stateFeatureSize = getCurrentState().stateFeatures.length;
		log(`Taille des fonctionnalités d'état déterminée: ${stateFeatureSize}`);
	}


	if (initialBestAgent && initialBestAgent.getActorWeights() && initialBestAgent.getCriticWeights()) {
		// Si un meilleur agent initial est fourni, l'utiliser comme le premier individu
		population.push(initialBestAgent);
		log("Population initialisée avec le meilleur agent chargé comme premier individu.");
		// Remplir le reste de la population avec des mutations de cet agent
		for (let i = 1; i < numIndividuals; i++) {
			let newActorWeights = JSON.parse(JSON.stringify(initialBestAgent.getActorWeights()));
			let newCriticWeights = JSON.parse(JSON.stringify(initialBestAgent.getCriticWeights()));
			newActorWeights = mutate(newActorWeights, mutationRate);
			newCriticWeights = mutate(newCriticWeights, mutationRate);
			population.push(new ActorCritic(
				learningRateActor, learningRateCritic, gamma,
				stateFeatureSize, NUM_ACTIONS, // Utiliser la variable pré-calculée
				newActorWeights, newCriticWeights
			));
		}
	} else {
		// Sinon, initialiser avec des agents aléatoires
		log("Population initialisée avec des agents aléatoires.");
		for (let i = 0; i < numIndividuals; i++) {
			population.push(new ActorCritic(
				learningRateActor,
				learningRateCritic,
				gamma,
				stateFeatureSize, // Utiliser la variable pré-calculée
				NUM_ACTIONS
			));
		}
	}

	// Définir la génération de départ pour cette exécution d'entraînement
	const startGeneration = currentGeneration; // Utilisez la génération globale actuelle

	// Modification de la boucle pour le mode infini
	let genCounter = startGeneration;
	while (isTraining && (isInfiniteMode || genCounter < startGeneration + numGenerations)) {
		currentGeneration = genCounter + 1; // Met à jour la variable globale de la génération actuelle
		currentGenerationSpan.textContent = currentGeneration;
		log(`--- Génération ${currentGeneration} ---`);

		let individualsWithFitness = [];
		let sumFitnessThisGen = 0;

		const tasksForWorkers = [];
		for (let i = 0; i < numIndividuals; i++) {
			const agent = population[i];
			const simulationParams = {
				numEpisodes: episodesPerIndividual,
				isTrainingRun: true,
				learningRateActor: learningRateActor,
				learningRateCritic: learningRateCritic,
				gamma: gamma,
				numStates: stateFeatureSize,
				numActions: NUM_ACTIONS,
				envConfig: { // Passer les paramètres d'environnement nécessaires au worker
					canvasWidth: canvas.width,
					canvasHeight: canvas.height,
					SQUARE_SIZE,
					SQUARE_SPEED,
					CIRCLE_RADIUS,
					MAX_CIRCLES,
					REWARD_COLLECT_CIRCLE,
					PENALTY_PER_STEP,
					BONUS_QUICK_COLLECTION_BASE,
					BONUS_QUICK_COLLECTION_DECAY,
					BONUS_PROXIMITY,
					PROXIMITY_THRESHOLD,
					BONUS_ACTION_VARIATION_PER_STEP,
					ACTION_HISTORY_BUFFER_SIZE,
					MIN_UNIQUE_ACTIONS_FOR_BONUS,
					BONUS_NO_WALL_HIT_EPISODE,
					PENALTY_WALL_HIT_PER_STEP,
					PENALTY_REPETITIVE_ACTION,
					REPETITIVE_ACTION_THRESHOLD_STEPS,
					PENALTY_EARLY_EMPTY_EPISODE,
					EARLY_END_TIME_THRESHOLD,
					COLLECTION_DISTANCE_THRESHOLD,
					PENALTY_STAGNATION,
					STAGNATION_THRESHOLD_STEPS,
					PENALTY_ACTION_CHANGE
				}
			};
			tasksForWorkers.push({ agentBrain: agent.saveBrain(), simulationParams: simulationParams });
		}

		// Utiliser le pool de workers pour évaluer tous les individus
		const results = await evaluateWithWorkerPool(tasksForWorkers);

		for (let i = 0; i < numIndividuals; i++) {
			const result = results[i]; // Les résultats sont déjà dans l'ordre grâce au pré-allocation

			if (result.status === 'fulfilled' && result.value) {
				const { fitness, agentBrain, circlesCollectedByIndividual } = result.value;
				const agent = population[i];
				agent.loadBrain(agentBrain); // Mettre à jour l'agent avec les poids potentiellement modifiés par l'entraînement du worker

				individualsWithFitness.push({ agent, fitness, circlesCollectedByIndividual });
				sumFitnessThisGen += fitness;

				// Limiter les logs pour ne pas spammer la console
				if (i % Math.max(1, Math.floor(numIndividuals / 5)) === 0) {
					log(`  Individu ${i + 1}/${numIndividuals}: Fitness = ${fitness.toFixed(2)} (Ronds: ${circlesCollectedByIndividual})`);
				}
			} else {
				console.error(`Erreur pour l'individu ${i}:`, result.reason);
				individualsWithFitness.push({ agent: population[i], fitness: -Infinity, circlesCollectedByIndividual: 0 });
			}

			if (!isTraining) {
				log("runGenerations: Sortie de l'évaluation individuelle car l'entraînement a été arrêté.");
				break;
			}
		}

		if (!isTraining) break;

		// Calculer la récompense moyenne de cette génération et mettre à jour le graphique
		const averageFitnessThisGen = sumFitnessThisGen / numIndividuals;
		updateLearningCurveChart(averageFitnessThisGen);
		log(`Récompense moyenne de la génération ${currentGeneration}: ${averageFitnessThisGen.toFixed(2)}`);


		individualsWithFitness.sort((a, b) => b.fitness - a.fitness);

		const bestFitnessThisGen = individualsWithFitness[0].fitness;
		const bestCirclesThisGen = individualsWithFitness[0].circlesCollectedByIndividual;

		log(`Meilleure fitness de la génération ${currentGeneration}: ${bestFitnessThisGen.toFixed(2)} (Ronds: ${bestCirclesThisGen})`);


		if (bestFitnessThisGen > bestRewardAllTime) {
			bestRewardAllTime = bestFitnessThisGen;
			bestCirclesAllTime = bestCirclesThisGen;
			bestRewardAllTimeSpan.textContent = bestRewardAllTime.toFixed(2);
			bestCirclesAllTimeSpan.textContent = bestCirclesAllTime;
			log(`Nouveau meilleur score global de la session ! Récompense : ${bestRewardAllTime.toFixed(2)} (${bestCirclesAllTime} ronds)`);


			bestActorWeights = individualsWithFitness[0].agent.getActorWeights();
			bestCriticWeights = individualsWithFitness[0].agent.getCriticWeights();

			bestAgentInstance = new ActorCritic(
				learningRateActor, learningRateCritic, gamma,
				stateFeatureSize, NUM_ACTIONS, // Utiliser la variable pré-calculée
				bestActorWeights, bestCriticWeights
			);
			log("Meilleure IA mise à jour, prête pour le test visuel.");

			// Sauvegarde automatique dans IndexedDB, y compris la génération actuelle et les données du graphique
			saveWeightsToIndexedDB(bestActorWeights, bestCriticWeights, bestRewardAllTime, bestCirclesAllTime, currentGeneration, learningCurveData, isInfiniteMode);
		}


		const nextPopulation = [];

		for (let i = 0; i < Math.min(numElitism, numIndividuals); i++) {
			nextPopulation.push(individualsWithFitness[i].agent);
		}


		const numParents = Math.ceil(numIndividuals * 0.5);
		const selectedParents = individualsWithFitness.slice(0, numParents);

		while (nextPopulation.length < numIndividuals) {
			if (!isTraining) {
				log("runGenerations: Arrêt de la création de la nouvelle population (entraînement interrompu).");
				break;
			}

			const parent1 = selectedParents[getRandomInt(0, selectedParents.length - 1)].agent;
			const parent2 = selectedParents[getRandomInt(0, selectedParents.length - 1)].agent;


			const parent1ActorWeights = parent1.getActorWeights();
			const parent1CriticWeights = parent1.getCriticWeights();
			const parent2ActorWeights = parent2.getActorWeights();
			const parent2CriticWeights = parent2.getCriticWeights();


			let childActorWeights = crossover(parent1ActorWeights, parent2ActorWeights);
			let childCriticWeights = crossover(parent1CriticWeights, parent2CriticWeights);


			childActorWeights = mutate(childActorWeights, mutationRate);
			childCriticWeights = mutate(childCriticWeights, mutationRate);


			nextPopulation.push(new ActorCritic(
				learningRateActor, learningRateCritic, gamma,
				stateFeatureSize, NUM_ACTIONS, // Utiliser la variable pré-calculée
				childActorWeights, childCriticWeights
			));
		}
		population = nextPopulation;

		genCounter++; // Incrémenter le compteur de générations pour la boucle while

		await new Promise(resolve => setTimeout(resolve, 0));
	}

	log(`runGenerations: Entraînement génétique terminé après ${currentGeneration} générations.`);
	stopTraining(false);
	isCanvasRenderingEnabled = true; // Réactiver le rendu du canvas après l'entraînement

	if (bestAgentInstance) {
		testBestIAButton.disabled = false;
	}
}

// --- Fonctions de dessin du jeu ---
function drawGame() {
	if (!isCanvasRenderingEnabled) return; // Ne pas dessiner si le rendu est désactivé

	ctx.clearRect(0, 0, canvas.width, canvas.height);

	// Dessine le carré
	ctx.fillStyle = '#007bff';
	ctx.fillRect(squareX, squareY, SQUARE_SIZE, SQUARE_SIZE);

	// Dessine les cercles
	ctx.fillStyle = '#00ff00';
	circles.forEach(circle => {
		ctx.beginPath();
		ctx.arc(circle.x, circle.y, CIRCLE_RADIUS, 0, Math.PI * 2);
		ctx.fill();
	});
}

// --- Test Visuel de la meilleure IA ---
async function visualTestGameLoop() {
	if (!isTestingVisual) {
		return;
	}


	if (!bestAgentInstance) {
		log("Erreur: L'agent de test n'est pas initialisé.");
		stopVisualTest();
		return;
	}

	// Optimisation: Appeler getCurrentState() une seule fois et passer l'état complet
	const currentStateData = getCurrentState();
	const { stateFeatures: state } = currentStateData;
	const actionIndex = bestAgentInstance.selectAction(state);

	const { reward: stepReward, collectedCount: stepCollectedCount } = stepEnvironmentAndGetReward(actionIndex, 0, currentStateData);

	// Met à jour les scores de test
	currentTestScoreSpan.textContent = (parseFloat(currentTestScoreSpan.textContent) + stepReward).toFixed(2);
	circlesCollectedTestSpan.textContent = parseInt(circlesCollectedTestSpan.textContent) + stepCollectedCount;

	drawGame();

	animationFrameId = requestAnimationFrame(visualTestGameLoop);
}

function startVisualTest() {
	if (!bestAgentInstance) {
		log("Aucune IA entraînée à tester visuellement. Lancez l'entraînement d'abord.");
		return;
	}
	if (isTraining) {
		log("Veuillez arrêter l'entraînement avant de lancer le test visuel.");
		return;
	}

	isTestingVisual = true;
	startButton.disabled = true;
	stopButton.disabled = true;
	testBestIAButton.disabled = true;
	stopTestButtonVisual.disabled = false;
	isCanvasRenderingEnabled = true; // S'assurer que le rendu est activé pour le test visuel

	resetGameEnvironment();
	currentTestScoreSpan.textContent = '0';
	circlesCollectedTestSpan.textContent = '0';

	log("Démarrage du test visuel de la meilleure IA...");
	visualTestGameLoop();
}

function stopVisualTest() {
	isTestingVisual = false;
	if (animationFrameId) {
		cancelAnimationFrame(animationFrameId);
		animationFrameId = null;
	}

	startButton.disabled = false;
	stopButton.disabled = true;
	testBestIAButton.disabled = false;
	stopTestButtonVisual.disabled = true;
	isCanvasRenderingEnabled = true; // S'assurer que le rendu est activé même après l'arrêt

	resetGameEnvironment();
	drawGame();
	log("Test visuel arrêté.");
}


// --- Gestion des événements UI ---

startButton.addEventListener('click', async () => {
	if (!isTraining) {
		isTraining = true;
		startButton.disabled = true;
		stopButton.disabled = false;
		settingsPanel.classList.remove('open');


		if (isTestingVisual) {
			cancelAnimationFrame(animationFrameId);
			animationFrameId = null;
			isTestingVisual = false;
			currentTestScoreSpan.textContent = '0';
			circlesCollectedTestSpan.textContent = '0';
			log("Test visuel interrompu au démarrage de l'entraînement génétique.");
		}

		testBestIAButton.disabled = true;
		stopTestButtonVisual.disabled = true;

		currentTestScoreSpan.textContent = '0';
		circlesCollectedTestSpan.textContent = '0';

		await new Promise(resolve => setTimeout(resolve, 50));
		// Passe l'instance du meilleur agent actuellement en mémoire (chargée ou de la dernière session)
		await runGenerations(bestAgentInstance);
	}
});

stopButton.addEventListener('click', () => {
	stopTraining(true);
});

function stopTraining(fromUser = false) {
	isTraining = false;
	startButton.disabled = false;
	stopButton.disabled = true;

	log("Entraînement arrêté" + (fromUser ? " par l'utilisateur." : " (terminé automatiquement)."));

	isCanvasRenderingEnabled = true; // Réactiver le rendu du canvas après l'entraînement

	if (bestAgentInstance) {
		testBestIAButton.disabled = false;
	}
	stopTestButtonVisual.disabled = true;
}

testBestIAButton.addEventListener('click', startVisualTest);
stopTestButtonVisual.addEventListener('click', stopVisualTest);

settingsButton.addEventListener('click', () => {
	settingsPanel.classList.toggle('open');
});

// Événement pour le checkbox du mode infini
infiniteGenerationsCheckbox.addEventListener('change', () => {
	numGenerationsInput.disabled = infiniteGenerationsCheckbox.checked;
	if (infiniteGenerationsCheckbox.checked) {
		log("Mode Générations Infinies activé. Le nombre de générations est ignoré.");
	} else {
		log("Mode Générations Infinies désactivé.");
	}
});

// --- Fonctions de sauvegarde/chargement via fichier ---

saveButton.addEventListener('click', () => {
	if (bestActorWeights && bestCriticWeights) {
		const dataToSave = {
			actorWeights: bestActorWeights,
			criticWeights: bestCriticWeights,
			bestReward: bestRewardAllTime,
			bestCircles: bestCirclesAllTime,
			currentGeneration: currentGeneration, // Sauvegarde la génération actuelle
			learningCurve: learningCurveData, // Sauvegarde les données du graphique
			infiniteGenerationsActive: infiniteGenerationsCheckbox.checked, // Sauvegarde l'état du mode infini
			numIndividuals: parseInt(numIndividualsInput.value),
			numGenerations: parseInt(numGenerationsInput.value),
			episodesPerIndividual: parseInt(episodesPerIndividualInput.value),
			mutationRate: parseFloat(mutationRateInput.value),
			learningRateActor: parseFloat(learningRateActorInput.value),
			learningRateCritic: parseFloat(learningRateCriticInput.value),
			gamma: parseFloat(gammaInput.value),
			numElitism: parseInt(numElitismInput.value),
			collectionDistanceThreshold: parseFloat(collectionDistanceThresholdInput.value),
			stagnationPenalty: parseFloat(stagnationPenaltyInput.value),
			stagnationThresholdSteps: parseInt(stagnationThresholdStepsInput.value),
			actionChangePenalty: parseFloat(actionChangePenaltyInput.value),
			timestamp: new Date().toISOString()
		};
		const jsonString = JSON.stringify(dataToSave, null, 2);
		const blob = new Blob([jsonString], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `actor_critic_agent_` + currentGeneration + `_gen.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
		log("Configuration de l'agent sauvegardée dans un fichier JSON.");
	} else {
		log("Aucun agent entraîné à sauvegarder. Veuillez lancer l'entraînement d'abord.");
	}
});

loadButton.addEventListener('click', () => {
	loadFileInput.click(); // Déclenche le clic sur l'input de fichier caché
});

loadFileInput.addEventListener('change', (event) => {
	const file = event.target.files[0];
	if (!file) {
		return;
	}

	const reader = new FileReader();
	reader.onload = async (e) => {
		try {
			const loadedData = JSON.parse(e.target.result);

			// Charger les poids
			if (loadedData.actorWeights && loadedData.criticWeights) {
				bestActorWeights = loadedData.actorWeights;
				bestCriticWeights = loadedData.criticWeights;

				// Mettre à jour les statistiques globales
				bestRewardAllTime = loadedData.bestReward || 0;
				bestCirclesAllTime = loadedData.bestCircles || 0;
				currentGeneration = loadedData.currentGeneration || 0; // Charge la génération depuis le fichier

				// Charger les données de la courbe d'apprentissage
				learningCurveData = loadedData.learningCurve || [];
				// Mettre à jour le graphique avec toutes les données chargées
				initLearningCurveChart(); // Réinitialiser pour un nouveau tracé complet
				Plotly.restyle(learningCurveChartDiv, { y: [learningCurveData] }); // Mettre à jour les données

				// Charger l'état du mode infini
				const loadedInfiniteMode = loadedData.infiniteGenerationsActive || false;
				infiniteGenerationsCheckbox.checked = loadedInfiniteMode;
				numGenerationsInput.disabled = loadedInfiniteMode;

				// S'assurer que stateFeatureSize est correct avant de créer le nouvel agent
				// Si la taille d'état n'a pas encore été définie, la définir ici
				if (stateFeatureSize === 0) {
					stateFeatureSize = getCurrentState().stateFeatures.length;
				}

				bestAgentInstance = new ActorCritic(
					parseFloat(learningRateActorInput.value),
					parseFloat(learningRateCriticInput.value),
					parseFloat(gammaInput.value),
					stateFeatureSize, // Utiliser la variable pré-calculée
					8, // S'assurer que le nombre d'actions est correct lors du chargement
					bestActorWeights,
					bestCriticWeights
				);
				log("Poids de l'agent chargés depuis le fichier.");

				bestRewardAllTimeSpan.textContent = bestRewardAllTime.toFixed(2);
				bestCirclesAllTimeSpan.textContent = bestCirclesAllTime;
				currentGenerationSpan.textContent = `${currentGeneration} (Chargé)`; // Indique la génération chargée

				// Mettre à jour les paramètres de l'UI si présents dans le fichier
				if (loadedData.numIndividuals) numIndividualsInput.value = loadedData.numIndividuals;
				if (loadedData.numGenerations) numGenerationsInput.value = loadedData.numGenerations;
				if (loadedData.episodesPerIndividual) episodesPerIndividualInput.value = loadedData.episodesPerIndividual;
				if (loadedData.mutationRate) mutationRateInput.value = loadedData.mutationRate;
				if (loadedData.learningRateActor) learningRateActorInput.value = loadedData.learningRateActor;
				if (loadedData.learningRateCritic) learningRateCriticInput.value = loadedData.learningRateCritic;
				if (loadedData.gamma) gammaInput.value = loadedData.gamma;
				if (loadedData.numElitism) numElitismInput.value = loadedData.numElitism;
				if (loadedData.collectionDistanceThreshold) collectionDistanceThresholdInput.value = loadedData.collectionDistanceThreshold;
				if (loadedData.stagnationPenalty) stagnationPenaltyInput.value = loadedData.stagnationPenalty;
				if (loadedData.stagnationThresholdSteps) stagnationThresholdStepsInput.value = loadedData.stagnationThresholdSteps;
				if (loadedData.actionChangePenalty) actionChangePenaltyInput.value = loadedData.actionChangePenalty;

				updateRewardParameters(); // S'assurer que les variables globales sont synchronisées

				testBestIAButton.disabled = false; // Permet de tester l'IA chargée
				stopTestButtonVisual.disabled = true; // Pas de test en cours
				startButton.disabled = false; // Permet de démarrer un nouvel entraînement
				stopButton.disabled = true; // Pas d'entraînement en cours

				// Effacer les données d'IndexedDB pour éviter des conflits si l'utilisateur charge et oublie
				await clearIndexedDB();
				log("IndexedDB effacé après le chargement depuis le fichier pour éviter les conflits de chargement automatique.");

			} else {
				log("Données d'agent invalides ou incomplètes dans le fichier.");
			}
		} catch (e) {
			console.error("Erreur lors de l'analyse ou du chargement du fichier :", e);
			log("Échec du chargement de l'agent à partir du fichier. Assurez-vous que c'est un JSON valide.");
		}
	};
	reader.readAsText(file);
});

resetButton.addEventListener('click', async () => {
	const confirmReset = await showCustomConfirm("Êtes-vous sûr de vouloir RÉINITIALISER complètement toutes les données et paramètres d'entraînement ? Cela effacera également les données sauvegardées dans le stockage local de votre navigateur.");
	if (confirmReset) {
		isTraining = false;
		isTestingVisual = false;
		if (animationFrameId) {
			cancelAnimationFrame(animationFrameId);
			animationFrameId = null;
		}

		// Réinitialiser les variables d'état d'entraînement globales
		bestRewardAllTime = 0;
		bestCirclesAllTime = 0;
		bestActorWeights = null;
		bestCriticWeights = null;
		bestAgentInstance = null;
		currentGeneration = 0; // Réinitialise la génération à 0
		learningCurveData = []; // Réinitialise les données du graphique
		resetLearningCurveChart(); // Réinitialise le graphique
		stateFeatureSize = 0; // Réinitialise la taille des fonctionnalités d'état

		// Réinitialiser les éléments de l'interface utilisateur aux valeurs par défaut
		currentGenerationSpan.textContent = '0';
		bestRewardAllTimeSpan.textContent = '0';
		bestCirclesAllTimeSpan.textContent = '0';
		currentTestScoreSpan.textContent = '0';
		circlesCollectedTestSpan.textContent = '0';

		// Réinitialiser les champs de saisie de l'interface utilisateur à leurs valeurs initiales
		numIndividualsInput.value = 10;
		numGenerationsInput.value = 100;
		numGenerationsInput.disabled = false; // Réactiver l'input
		infiniteGenerationsCheckbox.checked = false; // Décocher le checkbox
		episodesPerIndividualInput.value = 50;
		mutationRateInput.value = 0.05;
		learningRateActorInput.value = 0.001;
		learningRateCriticInput.value = 0.005;
		gammaInput.value = 0.99;
		numElitismInput.value = 1;
		collectionDistanceThresholdInput.value = 10;
		stagnationPenaltyInput.value = -0.2;
		stagnationThresholdStepsInput.value = 30;
		actionChangePenaltyInput.value = -0.02;

		document.getElementById('bonusProximity').value = 0.2;
		document.getElementById('proximityThreshold').value = 120;
		document.getElementById('bonusActionVariation').value = 0.005;
		document.getElementById('bonusNoWallHit').value = 1;
		document.getElementById('penaltyPerStep').value = -0.01;
		document.getElementById('penaltyWallHit').value = -0.5;
		document.getElementById('penaltyRepetitiveAction').value = -0.03;
		document.getElementById('penaltyEarlyEnd').value = -5;

		updateRewardParameters(); // Synchronise les variables globales avec les valeurs par défaut de l'UI

		// Réinitialiser les états des boutons
		startButton.disabled = false;
		stopButton.disabled = true;
		testBestIAButton.disabled = true;
		stopTestButtonVisual.disabled = true;

		// Effacer IndexedDB
		await clearIndexedDB();
		log("Toutes les données d'entraînement et le cache local ont été réinitialisés.");

		isCanvasRenderingEnabled = true; // S'assurer que le rendu est activé
		resetGameEnvironment();
		drawGame();
	}
});


// --- Initialisation au chargement de la fenêtre ---
window.onload = async function () {
	// Configuration initiale de la zone de jeu
	resetGameEnvironment();
	drawGame();

	// Initialiser le graphique de la courbe d'apprentissage
	initLearningCurveChart();

	// Initialise IndexedDB et tente de charger les données
	await openDatabase();
	const loadedData = await loadWeightsFromIndexedDB();
	if (loadedData) {
		bestActorWeights = loadedData.actorWeights;
		bestCriticWeights = loadedData.criticWeights;
		bestRewardAllTime = loadedData.bestReward || 0;
		bestCirclesAllTime = loadedData.bestCircles || 0;
		currentGeneration = loadedData.currentGeneration || 0; // Charge la génération au démarrage
		learningCurveData = loadedData.learningCurve || []; // Charge les données du graphique

		// Charger l'état du mode infini depuis IndexedDB
		const loadedInfiniteMode = loadedData.infiniteGenerationsActive || false;
		infiniteGenerationsCheckbox.checked = loadedInfiniteMode;
		numGenerationsInput.disabled = loadedInfiniteMode;

		// Définir la taille des fonctionnalités d'état après le chargement
		stateFeatureSize = getCurrentState().stateFeatures.length;

		bestAgentInstance = new ActorCritic(
			parseFloat(learningRateActorInput.value),
			parseFloat(learningRateCriticInput.value),
			parseFloat(gammaInput.value),
			stateFeatureSize, // Utiliser la variable pré-calculée
			8, // S'assurer que le nombre d'actions est correct lors du chargement
			bestActorWeights,
			bestCriticWeights
		);

		bestRewardAllTimeSpan.textContent = bestRewardAllTime.toFixed(2);
		bestCirclesAllTimeSpan.textContent = bestCirclesAllTime;
		currentGenerationSpan.textContent = `${currentGeneration} (Chargé)`; // Indique que c'est des données chargées

		testBestIAButton.disabled = false; // Active le bouton de test si un agent est chargé
		log("Session d'entraînement précédente chargée depuis IndexedDB.");
		// Pas besoin de Plotly.update ici, initLearningCurveChart sera appelé avec les données
		// Et Plotly.restyle dans le loadFileInput.addEventListener gère le chargement depuis fichier
	} else {
		testBestIAButton.disabled = true; // Garde le bouton de test désactivé si aucun agent
		log("Aucune session d'entraînement précédente trouvée.");
	}

	updateRewardParameters(); // Initialise les paramètres de récompense au chargement
	log("Prêt à commencer l'entraînement génétique.");
	log("Le jeu est prêt. Lancez l'entraînement pour qu'une IA soit disponible pour le test visuel.");

	initializeWorkers(); // Initialiser les web workers au chargement
};

