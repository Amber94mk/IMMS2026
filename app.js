/**
 * ===========================================================================
 *  Handwritten Digit Recognizer
 * ===========================================================================
 *
 * TASK TYPE: MULTI-CLASS CLASSIFICATION (supervised learning)
 *   Input : a 28 x 28 grayscale image of a handwritten digit
 *   Output: one of 10 classes (0, 1, ..., 9), plus a probability for each class
 *
 * METHOD: a CONVOLUTIONAL NEURAL NETWORK (CNN), a deep learning model
 *   designed for images. It learns useful visual features (edges, curves,
 *   loops) by itself instead of relying on hand-crafted rules.
 *
 * FULL WORKFLOW IN THIS PROGRAM
 *   STEP 1  Define the problem       -> 10-class classification (above)
 *   STEP 2  Prepare the data         -> mnist-data.js (download, normalize, split)
 *   STEP 3  Design the model         -> createModel()
 *   STEP 4  Choose how to learn      -> loss, optimizer, metric (in createModel)
 *   STEP 5  Train                    -> trainNewModel(): model.fit(...)
 *   STEP 6  Evaluate                 -> trainNewModel(): model.evaluate(...)
 *   STEP 7  Save and reuse the model -> model.save(...) / tf.loadLayersModel(...)
 *   STEP 8  Predict on new input     -> preprocessDrawing() + predict()
 */

// Where the trained model is saved inside the browser (IndexedDB),
// so it does not need to be retrained every time the page opens.
const MODEL_STORAGE_KEY = 'indexeddb://digit-recognizer-mnist-cnn';

// ---------- Training HYPERPARAMETERS ----------
// Hyperparameters are settings chosen by a person BEFORE training.
// (Parameters, in contrast, are the weights the model learns by itself.)
//
// EPOCH: one full pass over all 55,000 training images.
//   More epochs = more learning, but too many can cause OVERFITTING
//   (memorizing the training images instead of learning general patterns).
const EPOCHS = 2;
// BATCH SIZE: how many images the model looks at before updating its weights once.
//   55,000 / 256 = about 215 updates (ITERATIONS) per epoch, 430 in total.
//   Larger batches = smoother but fewer updates; smaller = noisier but more updates.
const BATCH_SIZE = 256;
// Stroke width on the 280x280 drawing canvas. After shrinking to 28x28 this
// gives a line thickness similar to MNIST handwriting.
const BRUSH_WIDTH = 22;

// ---------- DOM elements ----------
const statusEl = document.getElementById('status');
const progressBarEl = document.getElementById('progress-bar');
const drawCanvas = document.getElementById('draw-canvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('preview-canvas');
const previewCtx = previewCanvas.getContext('2d');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const barsEl = document.getElementById('bars');
const clearBtn = document.getElementById('clear-btn');
const retrainBtn = document.getElementById('retrain-btn');

let model = null;

// =====================================================================
// Status helpers
// =====================================================================

function setStatus(message) {
  statusEl.textContent = message;
}

function setProgress(fraction) {
  progressBarEl.style.width = `${Math.round(fraction * 100)}%`;
}

// =====================================================================
// STEP 3 + STEP 4: Design the model and choose how it learns
// =====================================================================

/**
 * Builds a small CNN for 28x28 grayscale images.
 *
 * Data flow and tensor shape after each layer (for one image):
 *
 *   Input            28 x 28 x 1
 *   Conv2D (32, 3x3) 26 x 26 x 32   params: (3*3*1 + 1) * 32  =     320
 *   MaxPooling (2x2) 13 x 13 x 32   params: 0
 *   Conv2D (64, 3x3) 11 x 11 x 64   params: (3*3*32 + 1) * 64 =  18,496
 *   MaxPooling (2x2)  5 x  5 x 64   params: 0
 *   Flatten          1600           params: 0
 *   Dropout (25%)    1600           params: 0
 *   Dense (128)      128            params: 1600*128 + 128    = 204,928
 *   Dense (10)       10             params: 128*10 + 10       =   1,290
 *                                   ---------------------------------------
 *                                   TOTAL trainable parameters  225,034
 *
 * The first part (Conv + Pooling) is the FEATURE EXTRACTOR.
 * The last part (Dense layers) is the CLASSIFIER that makes the decision.
 */
function createModel() {
  const m = tf.sequential(); // layers are stacked one after another

  // CONVOLUTION LAYER 1
  // 32 small 3x3 FILTERS slide across the image. Each filter computes a
  // weighted sum of the 9 pixels under it, so it "lights up" where it finds
  // a certain pattern (e.g. a vertical edge or a diagonal stroke).
  // The filter weights are NOT designed by hand; they are learned in training.
  //
  // ReLU ACTIVATION: relu(x) = max(0, x). It keeps positive signals and sets
  // negative ones to 0. This non-linearity lets the network learn complex shapes.
  m.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));

  // MAX POOLING 1
  // Splits the feature map into 2x2 blocks and keeps only the largest value.
  // This halves the width and height (less computation) and makes the model
  // tolerant to small shifts in where a stroke appears.
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // CONVOLUTION LAYER 2 + MAX POOLING 2
  // Works on the feature maps from layer 1, so it combines simple features
  // into more complex ones (e.g. edges -> curves -> loops like in 0, 6, 8, 9).
  m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // FLATTEN: turns the 5 x 5 x 64 feature maps into one list of 1,600 numbers
  // so they can be fed into ordinary fully connected (Dense) layers.
  m.add(tf.layers.flatten());

  // DROPOUT (25%): during training only, randomly switches off 25% of the
  // values at each step. The network cannot rely on any single feature, which
  // reduces OVERFITTING. Dropout is automatically disabled during prediction.
  m.add(tf.layers.dropout({ rate: 0.25 }));

  // DENSE (FULLY CONNECTED) LAYER: every one of the 1,600 inputs is connected
  // to each of the 128 neurons. It combines all features to judge the digit.
  m.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // OUTPUT LAYER: 10 neurons, one per digit.
  // SOFTMAX converts the 10 raw scores into probabilities that are all
  // between 0 and 1 and add up to 1 (e.g. "7: 92%, 1: 5%, ...").
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  // ---- STEP 4: how the model learns ----
  m.compile({
    // OPTIMIZER: the rule for updating the weights.
    // Adam is an improved version of gradient descent that adapts the step
    // size for each weight automatically (default learning rate 0.001).
    optimizer: tf.train.adam(),

    // LOSS FUNCTION: a single number that says "how wrong" the model is.
    // Categorical cross-entropy = -log(probability given to the correct digit).
    //   correct digit gets 0.9 probability -> loss = -ln(0.9) = 0.105 (small)
    //   correct digit gets 0.1 probability -> loss = -ln(0.1) = 2.303 (large)
    // Training = adjusting the weights to make this number as small as possible.
    loss: 'categoricalCrossentropy',

    // METRIC: shown to humans, not used for learning.
    // Accuracy = (number of correct predictions) / (number of images).
    metrics: ['accuracy'],
  });
  return m;
}

// =====================================================================
// STEP 5 + STEP 6 + STEP 7: Train, evaluate, save
// =====================================================================

/**
 * Downloads MNIST, trains a new model, reports test accuracy,
 * and saves the model to browser storage.
 */
async function trainNewModel() {
  setProgress(0);
  setStatus('Downloading MNIST dataset (about 11 MB)...');
  const data = new MnistData();
  await data.load();                       // STEP 2 (see mnist-data.js)

  const newModel = createModel();          // STEP 3 + 4
  const train = data.getTrainData();
  const test = data.getTestData();

  const batchesPerEpoch = Math.ceil(train.xs.shape[0] / BATCH_SIZE);
  const totalBatches = batchesPerEpoch * EPOCHS;
  let batchesDone = 0;
  let currentEpoch = 1;

  // ---- STEP 5: TRAINING ----
  // model.fit() repeats the following loop for every batch of 256 images:
  //   1. FORWARD PASS : run the images through the network -> 10 probabilities each
  //   2. LOSS         : compare the probabilities with the one-hot labels
  //                     (cross-entropy) -> one "wrongness" number
  //   3. BACKWARD PASS (BACKPROPAGATION): calculate, for each of the 225,034
  //                     weights, which direction would reduce the loss (the GRADIENT)
  //   4. UPDATE       : Adam moves every weight a small step in that direction
  // After about 215 batches one epoch is finished; the data is SHUFFLED and the
  // next epoch starts. Shuffling stops the model from learning the data order.
  setStatus(`Training model (epoch 1 of ${EPOCHS})...`);
  await newModel.fit(train.xs, train.ys, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    shuffle: true,
    callbacks: {
      // Callbacks let us watch training progress and update the screen.
      onEpochBegin: async (epoch) => {
        currentEpoch = epoch + 1;
      },
      onBatchEnd: async (batch, logs) => {
        batchesDone++;
        setProgress(batchesDone / totalBatches);
        if (batch % 10 === 0) {
          // The metric key is 'acc' in most TensorFlow.js versions.
          const accuracy = logs.acc ?? logs.accuracy ?? 0;
          setStatus(
            `Training model (epoch ${currentEpoch} of ${EPOCHS}) - ` +
            `batch accuracy ${(accuracy * 100).toFixed(1)}%`
          );
        }
      },
    },
  });

  // ---- STEP 6: EVALUATION ----
  // Measure accuracy on the 10,000 test images the model has NEVER seen.
  // High training accuracy alone is not enough: test accuracy shows whether
  // the model GENERALIZES to new handwriting.
  setStatus('Evaluating on test images...');
  const [lossTensor, accTensor] = newModel.evaluate(test.xs, test.ys, { batchSize: 1000 });
  const testAccuracy = (await accTensor.data())[0];

  // Free GPU memory used by the dataset (tensors are not garbage-collected).
  tf.dispose([train.xs, train.ys, test.xs, test.ys, lossTensor, accTensor]);

  // ---- STEP 7: SAVE ----
  // Saves the model structure and the learned weights so the next visit can
  // skip training. Saving can fail in some browsers (e.g. private mode);
  // the model still works for this session.
  let saveNote = '';
  try {
    await newModel.save(MODEL_STORAGE_KEY);
  } catch (err) {
    saveNote = ' (Could not save the model; it will retrain next time.)';
  }

  setProgress(1);
  setStatus(
    `Model ready. Test accuracy: ${(testAccuracy * 100).toFixed(2)}%. ` +
    `Draw a digit!${saveNote}`
  );
  return newModel;
}

/**
 * Tries to load a previously saved model; trains a new one if none exists.
 */
async function loadOrTrainModel() {
  try {
    const saved = await tf.loadLayersModel(MODEL_STORAGE_KEY);
    setProgress(1);
    setStatus('Loaded saved model from the browser. Draw a digit!');
    return saved;
  } catch (err) {
    // No saved model yet (first run) - train one.
    return trainNewModel();
  }
}

// =====================================================================
// Drawing on the canvas (collecting NEW input from the user)
// =====================================================================

let isDrawing = false;
let lastPoint = null;

function clearCanvas() {
  drawCtx.fillStyle = '#000000';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  predictionEl.textContent = '?';
  confidenceEl.innerHTML = '&nbsp;';
  renderBars(null);
}

/**
 * Converts a pointer event position into canvas pixel coordinates
 * (the canvas may be displayed smaller than its 280x280 resolution).
 */
function getCanvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function drawLine(from, to) {
  // White strokes on a black background, the same colors as MNIST.
  drawCtx.strokeStyle = '#ffffff';
  drawCtx.lineWidth = BRUSH_WIDTH;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
}

drawCanvas.addEventListener('pointerdown', (event) => {
  isDrawing = true;
  drawCanvas.setPointerCapture(event.pointerId);
  lastPoint = getCanvasPoint(event);
  drawLine(lastPoint, lastPoint); // draw a dot for a single click
});

drawCanvas.addEventListener('pointermove', (event) => {
  if (!isDrawing) return;
  const point = getCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
});

function stopDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  predict(); // recognize after each stroke
}

drawCanvas.addEventListener('pointerup', stopDrawing);
drawCanvas.addEventListener('pointercancel', stopDrawing);

// =====================================================================
// STEP 8a: Preprocessing - make the drawing look like an MNIST image
// =====================================================================

/**
 * A model only works well on data that looks like its training data.
 * MNIST digits are white on black, scaled to fit a 20x20 box, and centered by
 * their center of mass inside a 28x28 image. A digit drawn by the user may be
 * small, large, or off-center, so the SAME preparation is applied here.
 * (Skipping this step is a common reason for poor real-world accuracy.)
 *
 * Returns a Float32Array of 784 values in 0..1, or null if the canvas is empty.
 */
function preprocessDrawing() {
  const { width, height } = drawCanvas;
  const pixels = drawCtx.getImageData(0, 0, width, height).data;

  // 1. Find the BOUNDING BOX: the smallest rectangle containing all strokes.
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[(y * width + x) * 4] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // nothing drawn

  // 2. RESIZE: scale the digit so its longer side is 20 pixels, keeping the
  //    aspect ratio (a "1" stays thin), and place it in a 28x28 canvas.
  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = 20 / Math.max(boxW, boxH);
  const targetW = boxW * scale;
  const targetH = boxH * scale;

  const scaled = document.createElement('canvas');
  scaled.width = 28;
  scaled.height = 28;
  const scaledCtx = scaled.getContext('2d', { willReadFrequently: true });
  scaledCtx.fillStyle = '#000000';
  scaledCtx.fillRect(0, 0, 28, 28);
  scaledCtx.imageSmoothingEnabled = true;    // smooth shrinking gives soft gray
  scaledCtx.imageSmoothingQuality = 'high';  // edges, like real MNIST images
  scaledCtx.drawImage(
    drawCanvas,
    minX, minY, boxW, boxH,
    (28 - targetW) / 2, (28 - targetH) / 2, targetW, targetH
  );

  // 3. CENTER OF MASS: compute the brightness-weighted average position of the
  //    pixels and shift the image so that point lands at the center (14, 14).
  const small = scaledCtx.getImageData(0, 0, 28, 28).data;
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const value = small[(y * 28 + x) * 4];
      total += value;
      sumX += x * value;
      sumY += y * value;
    }
  }
  const shiftX = Math.round(14 - sumX / total);
  const shiftY = Math.round(14 - sumY / total);

  // The preview canvas shows the user exactly what the model receives.
  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, 28, 28);
  previewCtx.drawImage(scaled, shiftX, shiftY);

  // 4. NORMALIZE: read the final 28x28 pixels and divide by 255 -> 0..1,
  //    exactly as was done for the training data.
  const finalPixels = previewCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    input[i] = finalPixels[i * 4] / 255;
  }
  return input;
}

// =====================================================================
// STEP 8b: Prediction (INFERENCE) and result display
// =====================================================================

/**
 * Runs the trained model on the user's drawing.
 * Only the FORWARD PASS happens here; the weights do not change.
 */
function predict() {
  const input = preprocessDrawing();
  if (!input) return;

  if (!model) {
    confidenceEl.textContent = 'The model is not ready yet. Please wait.';
    return;
  }

  // The model expects a batch, so one image is shaped as [1, 28, 28, 1].
  // tf.tidy frees intermediate tensors automatically.
  const probabilities = tf.tidy(() => {
    const x = tf.tensor4d(input, [1, 28, 28, 1]);
    return model.predict(x).dataSync(); // 10 softmax probabilities
  });

  // DECISION RULE (ARGMAX): the predicted class is the digit with the
  // highest probability. That probability is shown as the "confidence".
  let best = 0;
  for (let digit = 1; digit < 10; digit++) {
    if (probabilities[digit] > probabilities[best]) best = digit;
  }

  predictionEl.textContent = best;
  confidenceEl.textContent = `Confidence: ${(probabilities[best] * 100).toFixed(1)}%`;
  renderBars(probabilities, best);
}

/**
 * Draws one horizontal bar per digit showing its probability.
 */
function renderBars(probabilities, best) {
  barsEl.innerHTML = '';
  for (let digit = 0; digit < 10; digit++) {
    const p = probabilities ? probabilities[digit] : 0;

    const row = document.createElement('div');
    row.className = 'bar-row' + (digit === best ? ' top' : '');
    row.innerHTML = `
      <span class="bar-label">${digit}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${p * 100}%"></div></div>
      <span class="bar-value">${(p * 100).toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  }
}

// =====================================================================
// Buttons and startup
// =====================================================================

clearBtn.addEventListener('click', clearCanvas);

retrainBtn.addEventListener('click', async () => {
  retrainBtn.disabled = true;
  try {
    const oldModel = model;
    model = null;
    if (oldModel) oldModel.dispose();
    model = await trainNewModel();
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    retrainBtn.disabled = false;
  }
});

async function main() {
  clearCanvas();
  if (typeof tf === 'undefined') {
    setStatus('Error: TensorFlow.js could not be loaded. Check your internet connection.');
    return;
  }

  retrainBtn.disabled = true;
  try {
    model = await loadOrTrainModel();
    predict(); // in case the user drew something while the model was loading
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    retrainBtn.disabled = false;
  }
}

main();
