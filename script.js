(function(ddc, tf) {
  const EXAMPLE_MP3 =
    "https://cdn.glitch.com/b0d48e64-a514-4d74-bcf7-8ebe00ad9ebb%2Ftest1_22050.mp3?v=1573149419598";
  const FEATURE_DIR =
    "https://dancedanceconvolution.com/assets/feature_extraction";
  const PLACEMENT_DIR =
    "https://dancedanceconvolution.com/assets/step_placement";
  const SELECTION_DIR =
    "https://dancedanceconvolution.com/assets/step_selection";

  function memoryUsageBytes() {
    return tf.memory().numBytes;
  }

  async function init() {
    console.log(`Allocated memory: ${memoryUsageBytes()}`);

    /*
    console.log("Loading audio");
    const audio = await ddc.audioIO.loadFromUrl(EXAMPLE_MP3);

    console.log("Extracting features");
    await ddc.featureExtraction.initialize(FEATURE_DIR);
    const features = await ddc.featureExtraction.extract(audio);

    console.log("Scoring audio");
    await ddc.stepPlacement.initialize(PLACEMENT_DIR);
    const placementScores = await ddc.stepPlacement.place(
      features,
      ddc.difficulty.MEDIUM
    );

    console.log("Finding peaks");
    const peaks = await ddc.stepPlacement.findPeaks(placementScores);
    const peaksThresholded = ddc.stepPlacement.thresholdPeaks(
      peaks,
      ddc.difficulty.MEDIUM
    );
    console.log(peaksThresholded);
    */
    const peaksThresholded = [7, 23, 31, 38];
    const timestamps = ddc.stepPlacement.peaksToTimestamps(peaksThresholded);

    console.log("Selecting steps");
    await ddc.stepSelection.initialize(SELECTION_DIR);
    const steps = await ddc.stepSelection.select(
      timestamps,
      ddc.difficulty.MEDIUM
    );

    await ddc.featureExtraction.dispose();
    await ddc.stepPlacement.dispose();
    await ddc.stepSelection.dispose();
    //features.dispose();
    //placementScores.dispose();

    console.log(`Allocated memory: ${memoryUsageBytes()}`);
  }

  init();
})(window.ddc, window.tf);
