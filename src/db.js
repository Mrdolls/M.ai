// db.js
// Gère les interactions avec IndexedDB pour la sauvegarde et le chargement des données.

const DB_NAME = 'ActorCriticTrainingDB';
const STORE_NAME = 'agentWeights';
const DB_VERSION = 1;
let db;

/**
 * Ouvre ou crée la base de données IndexedDB.
 * @returns {Promise<IDBDatabase>} La base de base de données IndexedDB.
 */
export function openDatabase() {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DB_NAME, DB_VERSION);

		request.onupgradeneeded = (event) => {
			db = event.target.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME, { keyPath: 'id' });
				console.log(`IndexedDB: Object store '${STORE_NAME}' created.`);
			}
		};

		request.onsuccess = (event) => {
			db = event.target.result;
			console.log("IndexedDB: Database opened successfully.");
			resolve(db);
		};

		request.onerror = (event) => {
			console.error("IndexedDB: Error opening database:", event.target.error);
			reject(event.target.error); // Rejeter explicitement en cas d'erreur d'ouverture
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
 * @param {Array<number>} learningCurveData - Données de la courbe d'apprentissage (y-values).
 * @param {Array<number>} learningCurveGenerations - Numéros de génération pour la courbe (x-values).
 * @param {number} plotlyUpdateFreq - La fréquence de mise à jour de Plotly.
 * @param {boolean} infiniteMode - État du mode infini.
 */
export async function saveWeightsToIndexedDB(actorWeights, criticWeights, reward, circlesCollected, currentGen, learningCurveData, learningCurveGenerations, plotlyUpdateFreq, infiniteMode) {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("IndexedDB: Failed to open database for saving:", error);
			return;
		}
	}

	try {
		const transaction = db.transaction([STORE_NAME], 'readwrite');
		const store = transaction.objectStore(STORE_NAME);

		const data = {
			id: 'bestAgent', // Clé unique pour stocker un seul meilleur agent
			actorWeights: actorWeights,
			criticWeights: criticWeights,
			bestReward: reward,
			bestCircles: circlesCollected,
			currentGeneration: currentGen,
			learningCurve: {
				data: learningCurveData,
				generations: learningCurveGenerations
			},
			plotlyUpdateFrequency: plotlyUpdateFreq, // Sauvegarde de la fréquence de mise à jour
			infiniteGenerationsActive: infiniteMode,
			timestamp: new Date().toISOString()
		};

		const request = store.put(data);

		request.onsuccess = () => {
			console.log("IndexedDB: Agent weights put request successful.");
		};

		request.onerror = (event) => {
			console.error("IndexedDB: Error on put request:", event.target.error);
		};

		transaction.oncomplete = () => {
			console.log("IndexedDB: Transaction for saving completed.");
		};

		transaction.onerror = (event) => {
			console.error("IndexedDB: Transaction error during saving:", event.target.error);
		};

		transaction.onabort = () => {
			console.warn("IndexedDB: Transaction for saving aborted.");
		};

	} catch (error) {
		console.error("IndexedDB: Error initiating save transaction:", error);
	}
}

/**
 * Charge les poids de l'acteur et du critique depuis IndexedDB.
 * @returns {Promise<object|null>} L'objet contenant les poids et autres données ou null si non trouvé.
 */
export async function loadWeightsFromIndexedDB() {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("IndexedDB: Failed to open database for loading:", error);
			return null;
		}
	}

	return new Promise((resolve) => {
		try {
			const transaction = db.transaction([STORE_NAME], 'readonly');
			const store = transaction.objectStore(STORE_NAME);
			const request = store.get('bestAgent');

			request.onsuccess = (event) => {
				const data = event.target.result;
				if (data) {
					console.log("IndexedDB: Agent weights loaded from DB.");
					resolve({
						actorWeights: data.actorWeights,
						criticWeights: data.criticWeights,
						bestReward: data.bestReward,
						bestCircles: data.bestCircles,
						currentGeneration: data.currentGeneration || 0,
						learningCurve: data.learningCurve ? data.learningCurve.data : [],
						learningCurveGenerations: data.learningCurve ? data.learningCurve.generations : [],
						plotlyUpdateFrequency: data.plotlyUpdateFrequency || 5, // Charge la fréquence, 5 par défaut
						infiniteGenerationsActive: data.infiniteGenerationsActive || false
					});
				} else {
					console.log("IndexedDB: No agent weights found in DB.");
					resolve(null);
				}
			};

			request.onerror = (event) => {
				console.error("IndexedDB: Error on get request:", event.target.error);
				resolve(null); // Résoudre à null en cas d'erreur de requête pour ne pas bloquer
			};

			transaction.oncomplete = () => {
				console.log("IndexedDB: Transaction for loading completed.");
			};

			transaction.onerror = (event) => {
				console.error("IndexedDB: Transaction error during loading:", event.target.error);
				resolve(null); // Résoudre à null en cas d'erreur de transaction pour ne pas bloquer
			};

			transaction.onabort = () => {
				console.warn("IndexedDB: Transaction for loading aborted.");
				resolve(null); // Résoudre à null si la transaction est annulée
			};

		} catch (error) {
			console.error("IndexedDB: Error initiating load transaction:", error);
			resolve(null); // Résoudre à null en cas d'erreur d'initialisation de transaction
		}
	});
}

/**
 * Efface toutes les données de la base de données IndexedDB.
 */
export async function clearIndexedDB() {
	if (!db) {
		try {
			db = await openDatabase();
		} catch (error) {
			console.error("IndexedDB: Failed to open database for clearing:", error);
			return;
		}
	}

	return new Promise((resolve) => {
		try {
			const transaction = db.transaction([STORE_NAME], 'readwrite');
			const store = transaction.objectStore(STORE_NAME);
			const request = store.clear();

			request.onsuccess = () => {
				console.log("IndexedDB: Clear request successful.");
			};

			request.onerror = (event) => {
				console.error("IndexedDB: Error on clear request:", event.target.error);
			};

			transaction.oncomplete = () => {
				console.log("IndexedDB: Transaction for clearing completed.");
				resolve();
			};

			transaction.onerror = (event) => {
				console.error("IndexedDB: Transaction error during clearing:", event.target.error);
				resolve(); // Résoudre même en cas d'erreur pour ne pas bloquer l'application
			};

			transaction.onabort = () => {
				console.warn("IndexedDB: Transaction for clearing aborted.");
				resolve(); // Résoudre même si la transaction est annulée
			};

		} catch (error) {
			console.error("IndexedDB: Error initiating clear transaction:", error);
			resolve();
		}
	});
}
