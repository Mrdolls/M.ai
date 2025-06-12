class Environment {
    constructor(gridSize, canvasId, numPoints) {
        this.gridSize = gridSize;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.numPoints = numPoints;
        this.cellSize = this.canvas.width / this.gridSize; // Assuming square canvas for now
        this.agentPos = { x: 0, y: 0 };
        this.points = [];
        this.score = 0;

        this.ACTION_SPACE = {
            UP: 0,
            DOWN: 1,
            LEFT: 2,
            RIGHT: 3,
            STAY: 4
        };
        this.actions = [this.ACTION_SPACE.UP, this.ACTION_SPACE.DOWN, this.ACTION_SPACE.LEFT, this.ACTION_SPACE.RIGHT, this.ACTION_SPACE.STAY];

        this.init();
    }

    init() {
        // Adjust canvas internal resolution to match its display size
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.cellSize = this.canvas.width / this.gridSize;

        this.agentPos = { x: Math.floor(this.gridSize / 2), y: Math.floor(this.gridSize / 2) };
        this.points = [];
        this.score = 0;
        this._placePoints();
        this.draw();
    }

    reset() {
        this.agentPos = { x: Math.floor(this.gridSize / 2), y: Math.floor(this.gridSize / 2) };
        this.points = [];
        this.score = 0;
        this._placePoints();
        // this.draw(); // Draw will be called by the main loop
        return this.getState();
    }

    _placePoints() {
        this.points = [];
        for (let i = 0; i < this.numPoints; i++) {
            let pointPos;
            do {
                pointPos = {
                    x: Math.floor(Math.random() * this.gridSize),
                    y: Math.floor(Math.random() * this.gridSize)
                };
            } while (this._isOccupied(pointPos) || (pointPos.x === this.agentPos.x && pointPos.y === this.agentPos.y));
            this.points.push(pointPos);
        }
    }

    _isOccupied(pos) {
        return this.points.some(p => p.x === pos.x && p.y === pos.y);
    }

    getState() {
        // Simple state: agent's x, y coordinates
        // More complex state could include relative positions of points
        // For Q-table, we need a string representation or a way to map this to an index
        let state = `agent_${this.agentPos.x}_${this.agentPos.y}`;
        // To keep the state space manageable with multiple points,
        // we could sort points by distance or use a fixed number of nearest points.
        // For now, let's try including the position of the first point (if any)
        // This will make the state space very large and might not be ideal for a simple Q-table.
        // A better approach for multiple points might be to encode the *relative* position of the *closest* point.

        // Let's refine state representation: relative position of the closest point.
        if (this.points.length > 0) {
            const closestPoint = this._getClosestPoint();
            if (closestPoint) {
                const relX = closestPoint.x - this.agentPos.x;
                const relY = closestPoint.y - this.agentPos.y;
                state += `_closest_${relX}_${relY}`;
            } else {
                 state += '_no_points';
            }
        } else {
            state += '_no_points';
        }
        return state;
    }

    _getClosestPoint() {
        if (this.points.length === 0) return null;
        let closest = this.points[0];
        let minDistance = Math.sqrt(Math.pow(this.agentPos.x - closest.x, 2) + Math.pow(this.agentPos.y - closest.y, 2));

        for (let i = 1; i < this.points.length; i++) {
            const dist = Math.sqrt(Math.pow(this.agentPos.x - this.points[i].x, 2) + Math.pow(this.agentPos.y - this.points[i].y, 2));
            if (dist < minDistance) {
                minDistance = dist;
                closest = this.points[i];
            }
        }
        return closest;
    }


    step(action) {
        let reward = -0.1; // Small penalty for each step to encourage efficiency
        let newPos = { ...this.agentPos };

        if (action === this.ACTION_SPACE.UP) newPos.y -= 1;
        else if (action === this.ACTION_SPACE.DOWN) newPos.y += 1;
        else if (action === this.ACTION_SPACE.LEFT) newPos.x -= 1;
        else if (action === this.ACTION_SPACE.RIGHT) newPos.x += 1;
        // No change for ACTION_SPACE.STAY

        let collectedPoint = false;
        let gameOver = false;

        // Check boundaries
        if (newPos.x < 0 || newPos.x >= this.gridSize || newPos.y < 0 || newPos.y >= this.gridSize) {
            reward = -1; // Penalty for hitting a wall
            // Agent stays in the same position if it hits a wall
        } else {
            this.agentPos = newPos;
        }

        // Check if agent collected a point
        const pointIndex = this.points.findIndex(p => p.x === this.agentPos.x && p.y === this.agentPos.y);
        if (pointIndex !== -1) {
            this.points.splice(pointIndex, 1);
            reward = 10; // Reward for collecting a point
            this.score++;
            collectedPoint = true;
            if (this.points.length === 0) {
                // gameOver = true; // Episode ends if all points collected
                // Let's not end the episode here, agent can continue exploring or new points can be added.
                // For now, collecting all points gives a big reward but episode continues until max steps.
                // This can be configured.
            }
        }

        // this.draw(); // Draw will be called by the main loop
        const nextState = this.getState();
        return { nextState, reward, done: gameOver, collectedPoint };
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid lines
        for (let i = 0; i <= this.gridSize; i++) {
            this.ctx.moveTo(i * this.cellSize, 0);
            this.ctx.lineTo(i * this.cellSize, this.canvas.height);
            this.ctx.moveTo(0, i * this.cellSize);
            this.ctx.lineTo(this.canvas.width, i * this.cellSize);
        }
        this.ctx.strokeStyle = '#ddd';
        this.ctx.stroke();

        // Draw points
        this.ctx.fillStyle = 'gold';
        this.points.forEach(p => {
            this.ctx.beginPath();
            this.ctx.arc(
                p.x * this.cellSize + this.cellSize / 2,
                p.y * this.cellSize + this.cellSize / 2,
                this.cellSize / 3, // Radius of the point
                0, 2 * Math.PI
            );
            this.ctx.fill();
        });

        // Draw agent
        this.ctx.fillStyle = 'blue';
        this.ctx.fillRect(
            this.agentPos.x * this.cellSize,
            this.agentPos.y * this.cellSize,
            this.cellSize,
            this.cellSize
        );
    }

    // Manual move for testing
    manualMove(action) {
        const result = this.step(action);
        console.log(`Manual move: ${action}, New State: ${result.nextState}, Reward: ${result.reward}`);
        this.draw();
        return result;
    }

    updateGridSize(newGridSize) {
        this.gridSize = newGridSize;
        // Canvas width/height should be clientWidth/Height for responsive sizing
        this.canvas.width = this.canvas.clientWidth;
        this.canvas.height = this.canvas.clientHeight;
        this.cellSize = this.canvas.width / this.gridSize;
        this.reset(); // Re-initialize points and agent position
    }

    updateNumPoints(newNumPoints) {
        this.numPoints = newNumPoints;
        this.reset(); // Re-initialize points
    }
}
