// --- Game Canvas and Context ---
// Get the HTML canvas element by its ID
const canvas = document.getElementById('gameCanvas');
// Get the 2D rendering context for the canvas, used for drawing
const ctx = canvas.getContext('2d');

// --- Game Configuration ---
const GRID_SIZE = 20; // The size (width and height in pixels) of each cell in the game grid
const MAP_WIDTH_CELLS = 20; // The width of the game map in terms of grid cells
const MAP_HEIGHT_CELLS = 20; // The height of the game map in terms of grid cells

// Set the canvas dimensions based on the map size and grid cell size
canvas.width = MAP_WIDTH_CELLS * GRID_SIZE;
canvas.height = MAP_HEIGHT_CELLS * GRID_SIZE;

// --- Game State Variables ---
let agent; // The AI-controlled agent object
let points = []; // An array to store active point objects on the map
let score = 0; // The current score of the game/episode
const MAX_POINTS = 5; // The maximum number of points allowed on the map simultaneously

// --- Agent Class ---
// Represents the AI-controlled entity that moves around the map.
class Agent {
    constructor(x = 0, y = 0, color = 'blue') {
        this.x = x; // The x-coordinate of the agent in grid cells
        this.y = y; // The y-coordinate of the agent in grid cells
        this.color = color; // The color to draw the agent
    }

    // Draws the agent on the canvas.
    draw() {
        ctx.fillStyle = this.color;
        ctx.fillRect(this.x * GRID_SIZE, this.y * GRID_SIZE, GRID_SIZE, GRID_SIZE);
    }

    // Moves the agent based on the given action and map boundaries.
    // Actions: 0: up, 1: down, 2: left, 3: right, 4: wait
    // Returns true if the agent's position changed, false otherwise.
    move(action) {
        const prevX = this.x;
        const prevY = this.y;

        switch (action) {
            case 0: // Up
                if (this.y > 0) this.y--;
                break;
            case 1: // Down
                if (this.y < MAP_HEIGHT_CELLS - 1) this.y++;
                break;
            case 2: // Left
                if (this.x > 0) this.x--;
                break;
            case 3: // Right
                if (this.x < MAP_WIDTH_CELLS - 1) this.x++;
                break;
            case 4: // Wait
                // Agent remains in the same position.
                break;
        }
        // Return true if position changed, false otherwise (e.g. hit a wall or chose to wait)
        return this.x !== prevX || this.y !== prevY;
    }

    // Returns the agent's current position.
    getPosition() {
        return { x: this.x, y: this.y };
    }
}

// --- Point Class ---
// Represents a collectible point on the map.
class Point {
    constructor(x, y, color = 'green') {
        this.x = x; // The x-coordinate of the point in grid cells
        this.y = y; // The y-coordinate of the point in grid cells
        this.color = color; // The color to draw the point
    }

    // Draws the point on the canvas as a circle.
    draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(
            this.x * GRID_SIZE + GRID_SIZE / 2, // Center x
            this.y * GRID_SIZE + GRID_SIZE / 2, // Center y
            GRID_SIZE / 3, // Radius
            0, // Start angle
            Math.PI * 2 // End angle
        );
        ctx.fill();
    }

    // Returns the point's position.
    getPosition() {
        return { x: this.x, y: this.y };
    }
}

// --- Game Functions ---

// Spawns a new point at a random, unoccupied location on the map.
function spawnPoint() {
    if (points.length >= MAX_POINTS) return; // Don't spawn if max points reached

    let newX, newY, overlapping;
    do {
        overlapping = false;
        newX = Math.floor(Math.random() * MAP_WIDTH_CELLS);
        newY = Math.floor(Math.random() * MAP_HEIGHT_CELLS);

        // Check for overlap with the agent's current position
        if (agent && agent.x === newX && agent.y === newY) {
            overlapping = true;
            continue;
        }
        // Check for overlap with existing points
        for (const point of points) {
            if (point.x === newX && point.y === newY) {
                overlapping = true;
                break;
            }
        }
    } while (overlapping); // Keep trying until a free spot is found
    points.push(new Point(newX, newY));
}

// Checks if the agent has collected any points.
// If a point is collected, increments score, spawns a new point, and removes the collected one.
// Returns true if a point was collected, false otherwise.
function checkPointCollection() {
    let collectedPoint = false;
    points = points.filter(point => {
        if (agent.x === point.x && agent.y === point.y) {
            score++;
            collectedPoint = true;
            // Update score display in the UI
            document.getElementById('currentScore').textContent = score;
            spawnPoint(); // Spawn a new point to replace the collected one
            return false; // Remove this point from the 'points' array
        }
        return true; // Keep this point
    });
    return collectedPoint;
}

// Resets the game to its initial state for a new episode.
// Places the agent randomly, clears existing points, resets score, and spawns initial points.
function resetGame() {
    agent = new Agent(
        Math.floor(Math.random() * MAP_WIDTH_CELLS),
        Math.floor(Math.random() * MAP_HEIGHT_CELLS)
    );
    points = [];
    score = 0;
    document.getElementById('currentScore').textContent = score; // Reset score in UI

    for (let i = 0; i < MAX_POINTS; i++) {
        spawnPoint();
    }
    drawGame(); // Redraw the game state
}

// Draws the entire game state on the canvas (grid, points, agent).
function drawGame() {
    // Clear the canvas before redrawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines (optional, for visual aid)
    ctx.strokeStyle = '#ddd'; // Light grey color for grid lines
    for (let i = 0; i <= MAP_WIDTH_CELLS; i++) {
        ctx.beginPath();
        ctx.moveTo(i * GRID_SIZE, 0);
        ctx.lineTo(i * GRID_SIZE, canvas.height);
        ctx.stroke();
    }
    for (let j = 0; j <= MAP_HEIGHT_CELLS; j++) {
        ctx.beginPath();
        ctx.moveTo(0, j * GRID_SIZE);
        ctx.lineTo(canvas.width, j * GRID_SIZE);
        ctx.stroke();
    }

    // Draw all active points
    points.forEach(point => point.draw());

    // Draw the agent
    if (agent) {
        agent.draw();
    }
}

// --- Main Game Step Function ---
// This function is called by the AI to advance the game by one time step based on an action.
// It moves the agent, checks for point collection, and calculates the reward.
function gameStep(action) {
    if (!agent) {
        console.error("Agent not initialized in gameStep");
        return { reward: 0, gameOver: true, newState: null, score: 0 };
    }

    // Default reward for taking a step (encourages efficiency)
    let reward = -0.1;
    agent.move(action); // Perform the action (move the agent)

    if (checkPointCollection()) {
        reward = 10; // Higher reward for collecting a point
    }

    // 'gameOver' for an episode is typically handled by the AI's training loop (e.g., max steps)
    // For this game, an individual step doesn't usually end the game itself.
    const gameOver = false;

    drawGame(); // Redraw the game after the step

    return {
        reward: reward,
        gameOver: gameOver, // This might be true if an episode-ending condition is met in game logic
        newState: agent.getPosition(), // The new state of the agent (its position)
        score: score // The current total score
    };
}

// --- Initialization ---
// Initial draw of the game board when the DOM is fully loaded.
document.addEventListener('DOMContentLoaded', () => {
    // Agent is created and game fully reset when training starts.
    // This initial drawGame will just show the empty grid.
    drawGame();
});

console.log("game.js loaded with comments.");
