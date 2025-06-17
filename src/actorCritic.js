// actorCritic.js
// Contient la définition de la classe ActorCritic, optimisée avec Float32Array.

export class ActorCritic {
	constructor(learningRateActor, learningRateCritic, gamma, numStates, numActions, initialActorWeights = null, initialCriticWeights = null) {
		this.lrActor = learningRateActor;
		this.lrCritic = learningRateCritic;
		this.gamma = gamma;
		this.numActions = numActions;
		this.numStates = numStates;

		// Initialisation des poids de l'acteur avec Float32Array
		if (initialActorWeights) {
			this.actorWeights = new Float32Array(initialActorWeights);
		} else {
			this.actorWeights = new Float32Array(numActions * numStates);
			for (let i = 0; i < this.actorWeights.length; i++) {
				this.actorWeights[i] = Math.random() * 0.2 - 0.1;
			}
		}

		// Initialisation des poids du critique avec Float32Array
		if (initialCriticWeights) {
			this.criticWeights = new Float32Array(initialCriticWeights);
		} else {
			this.criticWeights = new Float32Array(numStates);
			for (let i = 0; i < this.criticWeights.length; i++) {
				this.criticWeights[i] = Math.random() * 0.2 - 0.1;
			}
		}
	}

	predictCriticValue(stateFeatures) {
		let value = 0;
		for (let i = 0; i < stateFeatures.length; i++) {
			value += stateFeatures[i] * this.criticWeights[i];
		}
		return value;
	}

	predictActorScores(stateFeatures) {
		const scores = new Float32Array(this.numActions).fill(0);
		for (let action = 0; action < this.numActions; action++) {
			const baseIndex = action * this.numStates;
			for (let i = 0; i < stateFeatures.length; i++) {
				scores[action] += stateFeatures[i] * this.actorWeights[baseIndex + i];
			}
		}
		return scores;
	}

	softmax(scores) {
		const maxScore = Math.max(...scores);
		const expScores = scores.map(score => Math.exp(score - maxScore));
		const sumExpScores = expScores.reduce((sum, val) => sum + val, 0);
		return expScores.map(score => score / sumExpScores);
	}

	selectAction(stateFeatures) {
		const actionScores = this.predictActorScores(stateFeatures);
		const actionProbabilities = this.softmax(actionScores);

		let cumulativeProbability = 0;
		const r = Math.random();
		for (let i = 0; i < this.numActions; i++) {
			cumulativeProbability += actionProbabilities[i];
			if (r <= cumulativeProbability) {
				return i;
			}
		}
		return this.numActions - 1;
	}

	train(state, actionIndex, reward, nextState, done) {
		const currentV = this.predictCriticValue(state);
		const nextV = done ? 0 : this.predictCriticValue(nextState);
		const targetG = reward + this.gamma * nextV;
		const tdError = targetG - currentV;

		// Mise à jour des poids du critique
		for (let i = 0; i < state.length; i++) {
			this.criticWeights[i] += this.lrCritic * tdError * state[i];
		}

		// Mise à jour des poids de l'acteur
		const baseIndex = actionIndex * this.numStates;
		for (let i = 0; i < state.length; i++) {
			this.actorWeights[baseIndex + i] += this.lrActor * tdError * state[i];
		}
	}

	getActorWeights() {
		return new Float32Array(this.actorWeights); // Retourne une copie
	}

	setActorWeights(newWeights) {
		this.actorWeights = new Float32Array(newWeights);
	}

	getCriticWeights() {
		return new Float32Array(this.criticWeights); // Retourne une copie
	}

	setCriticWeights(newWeights) {
		this.criticWeights = new Float32Array(newWeights);
	}

	loadBrain(brain) {
		this.setActorWeights(brain.actorWeights);
		this.setCriticWeights(brain.criticWeights);
	}

	saveBrain() {
		return {
			actorWeights: this.getActorWeights(),
			criticWeights: this.getCriticWeights()
		};
	}
}