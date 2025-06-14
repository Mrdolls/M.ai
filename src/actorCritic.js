// actorCritic.js
// Contient la définition de la classe ActorCritic, utilisée par le thread principal et les Web Workers.

export class ActorCritic {
	constructor(learningRateActor, learningRateCritic, gamma, numStates, numActions, initialActorWeights = null, initialCriticWeights = null) {
		this.lrActor = learningRateActor;
		this.lrCritic = learningRateCritic;
		this.gamma = gamma;
		this.numActions = numActions;
		this.numStates = numStates;

		// Initialisation des poids de l'acteur
		if (initialActorWeights) {
			this.actorWeights = JSON.parse(JSON.stringify(initialActorWeights));
		} else {
			this.actorWeights = Array(numActions).fill(null).map(() =>
				Array(numStates).fill(null).map(() => Math.random() * 0.2 - 0.1)
			);
		}

		// Initialisation des poids du critique
		if (initialCriticWeights) {
			this.criticWeights = JSON.parse(JSON.stringify(initialCriticWeights));
		} else {
			this.criticWeights = Array(numStates).fill(null).map(() => Math.random() * 0.2 - 0.1);
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
		const scores = Array(this.numActions).fill(0);
		for (let action = 0; action < this.numActions; action++) {
			for (let i = 0; i < stateFeatures.length; i++) {
				scores[action] += stateFeatures[i] * this.actorWeights[action][i];
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

		for (let i = 0; i < state.length; i++) {
			this.criticWeights[i] += this.lrCritic * tdError * state[i];
		}

		for (let i = 0; i < state.length; i++) {
			this.actorWeights[actionIndex][i] += this.lrActor * tdError * state[i];
		}
	}

	getActorWeights() {
		return JSON.parse(JSON.stringify(this.actorWeights));
	}

	setActorWeights(newWeights) {
		this.actorWeights = JSON.parse(JSON.stringify(newWeights));
	}

	getCriticWeights() {
		return JSON.parse(JSON.stringify(this.criticWeights));
	}

	setCriticWeights(newWeights) {
		this.criticWeights = JSON.parse(JSON.stringify(newWeights));
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
