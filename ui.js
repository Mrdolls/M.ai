// This script handles interactions between the user interface (HTML elements)
// and the game/AI logic.

document.addEventListener('DOMContentLoaded', () => {
    // Get references to interactive HTML elements.
    const startButton = document.getElementById('startButton');
    const saveBrainButton = document.getElementById('saveBrainButton');

    // References to input fields (primarily for completeness, as ai.js reads them directly on start).
    // const learningRateInput = document.getElementById('learningRate');
    // const discountFactorInput = document.getElementById('discountFactor');
    // const explorationRateInput = document.getElementById('explorationRate');
    // const epsilonDecayInput = document.getElementById('epsilonDecay');
    // const trainingWavesInput = document.getElementById('trainingWaves');

    if (!startButton) {
        console.error("Start button not found in the DOM.");
        return; // Exit if critical elements are missing
    }
    if (!saveBrainButton) {
        console.error("Save Brain button not found in the DOM.");
        // Continue if only save button is missing, start might still work
    }

    // Event listener for the Start/Stop Training button.
    startButton.addEventListener('click', () => {
        // Check if the AI module and its functions are available.
        if (!window.ai || !window.ai.startTraining || !window.ai.stopTraining) {
            alert("Error: AI module not loaded correctly. Cannot start/stop training.");
            console.error("AI functions (startTraining/stopTraining) not found on window.ai.");
            return;
        }

        // Toggle between starting and stopping training based on button text.
        if (startButton.textContent === "Start Training") {
            console.log("UI: Attempting to start training.");
            window.ai.startTraining(); // Call AI function to start.
            startButton.textContent = "Stop Training";
            startButton.style.backgroundColor = '#e74c3c'; // Red color for "Stop"
        } else {
            console.log("UI: Attempting to stop training.");
            window.ai.stopTraining(); // Call AI function to stop.
            startButton.textContent = "Start Training";
            startButton.style.backgroundColor = '#3498db'; // Blue color for "Start"
        }
    });

    // Event listener for the Save Best Brain button.
    if (saveBrainButton) {
        saveBrainButton.addEventListener('click', () => {
            console.log("UI: Save Brain button clicked.");
            if (!window.ai || !window.ai.getBestBrain) {
                alert("Error: AI module not loaded correctly. Cannot save brain.");
                console.error("AI getBestBrain function not found on window.ai.");
                return;
            }

            const brain = window.ai.getBestBrain(); // Get the Q-table from ai.js.

            if (brain && Object.keys(brain).length > 0) {
                // Convert the Q-table (JavaScript object) to a JSON string.
                // The 'null, 2' arguments pretty-print the JSON with an indent of 2 spaces.
                const brainJson = JSON.stringify(brain, null, 2);

                // Create a Blob (Binary Large Object) with the JSON data.
                const blob = new Blob([brainJson], { type: 'application/json' });

                // Create a temporary anchor (<a>) element to trigger the download.
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob); // Create a URL for the blob.
                link.download = 'best_ai_q_table_brain.json'; // Suggested filename for the download.

                // Programmatically click the link to initiate the download.
                document.body.appendChild(link); // Link needs to be in the document to be clicked.
                link.click();
                document.body.removeChild(link); // Clean up by removing the temporary link.

                URL.revokeObjectURL(link.href); // Release the object URL to free resources.
                console.log("Best brain data prepared for download.");
            } else {
                alert("No 'best brain' has been recorded yet, or the brain is empty. Train the AI first!");
                console.warn("Attempted to save an empty or non-existent brain.");
            }
        });
    }

    // Any other UI initializations or dynamic updates could go here.
    // For example, dynamically populating dropdowns, or validating inputs on change.
});

console.log("ui.js loaded with comments.");
