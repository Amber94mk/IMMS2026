/**
 * ===========================================================================
 *  MNIST data loader  (STEP 2 of the machine learning workflow: prepare data)
 * ===========================================================================
 *
 * WHAT IS MNIST?
 *   MNIST is the classic "hello world" dataset for image classification.
 *   It contains handwritten digits (0-9) written by real people.
 *   - Each image is 28 x 28 pixels, grayscale (one brightness value per pixel).
 *   - Digits are white on a black background.
 *   - Every image has a LABEL: the correct answer (which digit it is).
 *   Because every example comes with the correct answer, this is
 *   SUPERVISED LEARNING: the model learns from (input, answer) pairs.
 *
 * HOW THE DATA IS STORED (Google's public TensorFlow.js example files):
 *   - mnist_images.png : one big image ("sprite"). Each ROW of 784 pixels
 *                        (= 28 * 28) is one flattened digit image.
 *                        65,000 rows = 65,000 images.
 *   - mnist_labels_uint8 : 10 bytes per image in ONE-HOT format.
 *                        Example: the digit 3 is stored as
 *                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
 *                        (a 1 only at position 3).
 *                        One-hot encoding lets the label be compared directly
 *                        with the model's 10 output probabilities.
 *
 * TRAIN / TEST SPLIT:
 *   - 55,000 images are used for TRAINING (the model learns from these).
 *   - 10,000 images are kept aside for TESTING (never shown during training).
 *   The test set measures GENERALIZATION: how well the model handles
 *   handwriting it has never seen, which is what matters for real users.
 */

const IMAGE_SIZE = 784;          // 28 * 28 pixels per image
const NUM_CLASSES = 10;          // 10 possible answers: digits 0..9
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS; // 10,000

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

class MnistData {
  /**
   * Downloads the image sprite and labels and stores them as typed arrays.
   */
  async load() {
    // --- 1) Download the sprite image -----------------------------------
    const img = new Image();
    img.crossOrigin = 'anonymous'; // required to read pixels from another domain

    const imageLoaded = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error('Failed to download MNIST images.'));
    });
    img.src = MNIST_IMAGES_SPRITE_PATH;
    await imageLoaded;

    // --- 2) Copy the pixels into a Float32Array -------------------------
    // A canvas is used as a tool to read pixel values from the image.
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const datasetBytes = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);

    // Read the sprite in chunks of 5,000 rows to avoid creating one huge canvas.
    const chunkSize = 5000;
    canvas.width = img.width;      // 784
    canvas.height = chunkSize;

    for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
      ctx.drawImage(
        img,
        0, i * chunkSize, img.width, chunkSize,   // source rectangle
        0, 0, img.width, chunkSize                // destination rectangle
      );
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Canvas pixels are stored as R, G, B, A (4 values per pixel).
      // The sprite is grayscale (R = G = B), so the red channel is enough.
      //
      // NORMALIZATION: pixel values 0..255 are divided by 255 to get 0..1.
      // Neural networks train faster and more stably when inputs are small
      // numbers on a common scale.
      const offset = i * chunkSize * IMAGE_SIZE;
      for (let j = 0; j < imageData.data.length / 4; j++) {
        datasetBytes[offset + j] = imageData.data[j * 4] / 255;
      }
    }

    // --- 3) Download the one-hot labels ---------------------------------
    const response = await fetch(MNIST_LABELS_PATH);
    if (!response.ok) {
      throw new Error('Failed to download MNIST labels.');
    }
    const labels = new Uint8Array(await response.arrayBuffer());

    // --- 4) Split into training and test sets ---------------------------
    // The first 55,000 images train the model; the last 10,000 test it.
    this.trainImages = datasetBytes.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = datasetBytes.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = labels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = labels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  /**
   * Returns the full training set as tensors.
   *
   * A TENSOR is a multi-dimensional array of numbers, the basic data type of
   * deep learning libraries.
   *   xs: shape [55000, 28, 28, 1] = [number of images, height, width, channels]
   *       (1 channel because the images are grayscale; color would be 3)
   *   ys: shape [55000, 10] = one-hot label for each image
   */
  getTrainData() {
    const xs = tf.tensor4d(this.trainImages, [NUM_TRAIN_ELEMENTS, 28, 28, 1]);
    const ys = tf.tensor2d(this.trainLabels, [NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
    return { xs, ys };
  }

  /**
   * Returns the full test set as tensors (used only to measure accuracy).
   *   xs: shape [10000, 28, 28, 1], ys: shape [10000, 10]
   */
  getTestData() {
    const xs = tf.tensor4d(this.testImages, [NUM_TEST_ELEMENTS, 28, 28, 1]);
    const ys = tf.tensor2d(this.testLabels, [NUM_TEST_ELEMENTS, NUM_CLASSES]);
    return { xs, ys };
  }
}
