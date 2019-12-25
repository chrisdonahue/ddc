(function(ddc) {
  const EXAMPLE_MP3 =
    "https://cdn.glitch.com/b0d48e64-a514-4d74-bcf7-8ebe00ad9ebb%2Ftest1_22050.mp3?v=1573149419598";
  const FEATURE_DIR = "https://chrisdonahue.com/ddc/assets/feature_extraction";
  const PLACEMENT_DIR = "https://chrisdonahue.com/ddc/assets/step_placement";
  const SELECTION_DIR = "https://chrisdonahue.com/ddc/assets/step_selection";

  async function init() {
    console.log("Loading audio");
    const audio = await ddc.audioIO.loadFromUrl(EXAMPLE_MP3);

    console.log("Extracting features");
    await ddc.featureExtraction.initialize(FEATURE_DIR);
    const features = await ddc.featureExtraction.extract(audio);

    console.log("Placing steps");
    await ddc.stepPlacement.initialize(PLACEMENT_DIR);
    const placementScores = await ddc.stepPlacement.place(
      features,
      ddc.stepPlacement.difficulty.MEDIUM
    );
    //console.log(placementScores);
    console.log(placementScores.dataSync());

    console.log("Selecting steps");

    features.dispose();
    placementScores.dispose();
  }

  init();
})(window.ddc);
