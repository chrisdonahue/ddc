/*

Dance Dance Convolution
Donahue et al. 2017

All code by Chris Donahue
Copyright 2019

chrisdonahue.com
twitter.com/chrisdonahuey
github.com/chrisdonahue

*/

window.ddc = window.ddc || {};

(function(ddc, AudioContext, tf) {
  const DEBUG = false;
  const DIFFICULTY_NAME_TO_ID = {
    BEGINNER: 0,
    EASY: 1,
    MEDIUM: 2,
    HARD: 3,
    CHALLENGE: 4
  };
  const DIFFICULTY_ID_TO_NAME = {
    0: "BEGINNER",
    1: "EASY",
    2: "MEDIUM",
    3: "HARD",
    4: "CHALLENGE"
  };

  ddc.difficulty = DIFFICULTY_NAME_TO_ID;

  /* Helper methods */

  async function retrieveVars(ckptDirUrl) {
    if (ckptDirUrl === undefined) {
      throw new Error("Checkpoint directory URL unspecified.");
    }

    const vars = await fetch(`${ckptDirUrl}/weights_manifest.json`)
      .then(response => response.json())
      .then(manifest => tf.io.loadWeights(manifest, ckptDirUrl));
    return vars;
  }

  async function dispose(vars) {
    Object.keys(vars).forEach(name => vars[name].dispose());
  }

  /* Audio IO module maps files to audio */

  const audioCtx = new AudioContext();

  async function loadFromFile(fileHandler) {
    throw new Error("undefined");
  }

  async function loadFromUrl(uri) {
    const result = await fetch(uri, { method: "GET" });
    const buffer = await audioCtx.decodeAudioData(await result.arrayBuffer());

    if (buffer.sampleRate !== 44100) {
      // TODO: Resample
      throw new Error("Invalid sample rate");
    }

    const monoAudio = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; ++i) {
      monoAudio[i] = 0;
    }
    for (let ch = 0; ch < buffer.numberOfChannels; ++ch) {
      const channel = buffer.getChannelData(ch);
      for (let i = 0; i < buffer.length; ++i) {
        monoAudio[i] += channel[i];
      }
    }
    for (let i = 0; i < buffer.length; ++i) {
      monoAudio[i] /= buffer.numberOfChannels;
    }

    // TODO: Return ID3 tags for artist/title if exists
    return monoAudio;
  }

  ddc.audioIO = {};
  ddc.audioIO.loadFromFile = loadFromFile;
  ddc.audioIO.loadFromUrl = loadFromUrl;
  ddc.audioIO.resample = null;

  /* Feature extraction module maps audio to spectrogram */

  const FFT_FRAME_LENGTHS = [1024, 2048, 4096];
  const FFT_FRAME_STEP = 441;
  const LOG_EPS = 1e-16;

  let featureModelVars = null;

  async function featureInitialize(ckptDirUrl) {
    if (featureModelVars !== null) {
      await featureDispose();
    }
    featureModelVars = await retrieveVars(ckptDirUrl);
  }

  async function featureDispose() {
    if (featureModelVars === null) return;
    await dispose(featureModelVars);
    featureModelVars = null;
  }

  async function featureExtract(waveformArr) {
    if (featureModelVars === null) {
      throw new Error("Must call initialize method first");
    }

    if (DEBUG) {
      return tf.ones([1, 80, 3], "float32");
    }

    const feats = tf.tidy(() => {
      const waveform = tf.tensor1d(waveformArr, "float32");

      let feats = [];
      let featsNumTimesteps = null;
      for (let i = 0; i < FFT_FRAME_LENGTHS.length; ++i) {
        // TODO: Inner tf.tidy() to clean up intermediate tensors and only store lmel spectrogram

        // Pad waveform to center spectrogram
        const fftFrameLength = FFT_FRAME_LENGTHS[i];
        const fftFrameLengthHalf = fftFrameLength / 2;
        const waveformPadded = tf.pad(waveform, [[fftFrameLengthHalf, 0]]);

        // Slice waveform into frames
        const waveformFrames = tf.signal.frame(
          waveformPadded,
          fftFrameLength,
          FFT_FRAME_STEP,
          true
        );

        // Apply window
        const window = featureModelVars[`window_bh_${fftFrameLength}`];
        let waveformFramesWindowed = tf.mul(waveformFrames, window);

        // Copying questionable "zero phase" windowing nonsense from essentia
        // https://github.com/MTG/essentia/blob/master/src/algorithms/standard/windowing.cpp#L85
        const waveformFramesHalfOne = tf.slice(
          waveformFramesWindowed,
          [0, 0],
          [waveformFrames.shape[0], fftFrameLengthHalf]
        );
        const waveformFramesHalfTwo = tf.slice(
          waveformFramesWindowed,
          [0, fftFrameLengthHalf],
          [waveformFrames.shape[0], fftFrameLengthHalf]
        );
        waveformFramesWindowed = tf.concat(
          [waveformFramesHalfTwo, waveformFramesHalfOne],
          1
        );

        // Perform FFT
        const spectrogram = tf.spectral.rfft(waveformFramesWindowed);

        // Compute magnitude spectrogram
        const magSpectrogram = tf.abs(spectrogram);

        // Computer power spectrogram
        const powSpectrogram = tf.square(magSpectrogram);

        // Compute mel spectrogram
        const spectrogramLength = fftFrameLengthHalf + 1;
        const melW = featureModelVars[`mel_${spectrogramLength}`];
        const melSpectrogram = tf.matMul(powSpectrogram, melW);

        // Compute log mel spectrogram
        const logMelSpectrogram = tf.log(
          tf.add(melSpectrogram, tf.scalar(LOG_EPS, "float32"))
        );

        // Add feats to array
        feats.push(logMelSpectrogram);

        // Find shortest analysis length
        if (
          featsNumTimesteps === null ||
          spectrogram.shape[0] < featsNumTimesteps
        ) {
          featsNumTimesteps = spectrogram.shape[0];
        }
      }

      // Trim excess padding
      for (let i = 0; i < feats.length; ++i) {
        if (feats[i].shape[0] > featsNumTimesteps) {
          feats[i] = tf.slice(
            feats[i],
            [0, 0],
            [featsNumTimesteps, feats[i].shape[1]]
          );
        }
      }

      // Stack features from different frame lengths as "channels"
      feats = tf.stack(feats, 2);

      return feats;
    });

    return feats;
  }

  ddc.featureExtraction = {};
  ddc.featureExtraction.initialize = featureInitialize;
  ddc.featureExtraction.dispose = featureDispose;
  ddc.featureExtraction.extract = featureExtract;

  /* Step placement module maps spectrogram to events */

  const THRESHOLDS = {
    BEGINNER: 0.15325437,
    EASY: 0.23268291,
    MEDIUM: 0.29456162,
    HARD: 0.29084727,
    CHALLENGE: 0.28875697
  };
  const FEATURE_RATE = 100;
  const FEATURE_CONTEXT_RADIUS = 7;

  let placementModelVars = null;

  async function placementInitialize(ckptDirUrl) {
    if (placementModelVars !== null) {
      await placementDispose();
    }
    placementModelVars = await retrieveVars(ckptDirUrl);
  }

  async function placementDispose() {
    if (placementModelVars === null) return;
    await dispose(placementModelVars);
    placementModelVars = null;
  }

  async function place(feats, difficulty) {
    if (placementModelVars === null) {
      throw new Error("Must call initialize method first");
    }
    if (
      feats.shape.length !== 3 ||
      feats.shape[1] !== 80 ||
      feats.shape[2] !== 3
    ) {
      throw new Error("Invalid feature dimensions");
    }
    if (difficulty < 0 || difficulty > 4) {
      throw new Error("Invalid difficulty specified");
    }

    const scores = tf.tidy(() => {
      // Normalize features to have zero mean and unit variance
      let featsNormalized = tf.sub(feats, placementModelVars["feats_mean"]);
      featsNormalized = tf.div(
        featsNormalized,
        placementModelVars["feats_std"]
      );

      // Pad features to make context slices
      const featsPadded = tf.pad(featsNormalized, [
        [FEATURE_CONTEXT_RADIUS, FEATURE_CONTEXT_RADIUS],
        [0, 0],
        [0, 0]
      ]);

      // Construct one-hot tensor from difficulty
      const difficultyOneHot = tf.expandDims(
        tf.cast(tf.oneHot(tf.scalar(difficulty, "int32"), 5), "float32"),
        0
      );

      let scores = [];
      for (let i = 0; i < feats.shape[0]; ++i) {
        // TODO: Batch

        // Create slice of spectrogram
        let x = tf.slice(
          featsPadded,
          [i, 0, 0],
          [FEATURE_CONTEXT_RADIUS * 2 + 1, feats.shape[1], feats.shape[2]]
        );
        x = x.expandDims(0);

        // Conv 0
        x = tf.conv2d(
          x,
          placementModelVars["model_sp/cnn_0/filters"],
          [1, 1],
          "valid"
        );
        x = tf.add(x, placementModelVars["model_sp/cnn_0/biases"]);
        x = tf.relu(x);
        x = tf.maxPool(x, [1, 3], [1, 3], "same");

        // Conv 1
        x = tf.conv2d(
          x,
          placementModelVars["model_sp/cnn_1/filters"],
          [1, 1],
          "valid"
        );
        x = tf.add(x, placementModelVars["model_sp/cnn_1/biases"]);
        x = tf.relu(x);
        x = tf.maxPool(x, [1, 3], [1, 3], "same");

        // Concat difficulty
        x = tf.reshape(x, [1, -1]);
        x = tf.concat([x, difficultyOneHot], 1);

        // Dense 0
        x = tf.matMul(x, placementModelVars["model_sp/dnn_0/W"]);
        x = tf.add(x, placementModelVars["model_sp/dnn_0/b"]);
        x = tf.relu(x);

        // Dense 1
        x = tf.matMul(x, placementModelVars["model_sp/dnn_1/W"]);
        x = tf.add(x, placementModelVars["model_sp/dnn_1/b"]);
        x = tf.relu(x);

        // Output
        x = tf.matMul(x, placementModelVars["model_sp/logit/W"]);
        x = tf.add(x, placementModelVars["model_sp/logit/b"]);
        x = tf.sigmoid(x);
        x = tf.reshape(x, [1]);

        scores.push(x);
      }

      scores = tf.concat(scores, 0);

      return scores;
    });

    return scores;
  }

  function findPeaks(scores) {
    const peaks = tf.tidy(() => {
      const window = tf.signal.hammingWindow(5);
      let smoothedScores = tf.conv1d(
        tf.reshape(scores, [-1, 1]),
        tf.reshape(window, [-1, 1, 1]),
        1,
        "same"
      );
      smoothedScores = smoothedScores.dataSync();
      const peaks = [];
      for (let i = 0; i < smoothedScores.length; ++i) {
        const l = i === 0 ? 0 : smoothedScores[i - 1];
        const c = smoothedScores[i];
        const r = i === smoothedScores.length - 1 ? 0 : smoothedScores[i + 1];
        if (c >= l && c >= r) {
          peaks.push(i);
        }
      }
      return peaks;
    });
    return peaks;
  }

  function thresholdPeaks(peaks, difficulty, threshold) {
    if (difficulty === undefined && threshold === undefined) {
      console.log("Must specify difficulty or threshold");
    }
    if (difficulty !== undefined) {
      threshold = THRESHOLDS[DIFFICULTY_ID_TO_NAME[difficulty]];
    }
    const thresholded = [];
    for (let i = 0; i < peaks.length; ++i) {
      if (peaks[i] >= threshold) {
        thresholded.push(peaks[i]);
      }
    }
    return thresholded;
  }

  function peaksToTimestamps(peaks) {
    const timestamps = [];
    for (let i = 0; i < peaks.length; ++i) {
      timestamps.push(peaks[i] / FEATURE_RATE);
    }
    return timestamps;
  }

  ddc.stepPlacement = {};
  ddc.stepPlacement.thresholds = THRESHOLDS;
  ddc.stepPlacement.initialize = placementInitialize;
  ddc.stepPlacement.dispose = placementDispose;
  ddc.stepPlacement.place = place;
  ddc.stepPlacement.findPeaks = findPeaks;
  ddc.stepPlacement.thresholdPeaks = thresholdPeaks;
  ddc.stepPlacement.peaksToTimestamps = peaksToTimestamps;

  /* Step selection module maps timestamped events to choreography */

  const VOCAB_NUM_SPECIAL = 1;
  const VOCAB_NUM_ARROW_DIRECTIONS = 4;
  const VOCAB_NUM_ARROW_TYPES = 4;
  const VOCAB_SIZE =
    VOCAB_NUM_SPECIAL +
    Math.pow(VOCAB_NUM_ARROW_DIRECTIONS, VOCAB_NUM_ARROW_TYPES);
  const RNN_NUM_LAYERS = 2;
  const RNN_CELL_SIZE = 128;

  let selectionModelVars = null;
  let selectionLSTMCells = [];
  let selectionLSTMForgetBias = null;

  async function selectionInitialize(ckptDirUrl) {
    if (selectionModelVars !== null) {
      await selectionDispose();
    }
    selectionModelVars = await retrieveVars(ckptDirUrl);
    selectionLSTMForgetBias = tf.scalar(1, "float32");
    for (let i = 0; i < RNN_NUM_LAYERS; ++i) {
      const layerPrefix = `model_ss/rnn_unroll/MultiRNNCell/Cell${i}/BasicLSTMCell/Linear`;
      selectionLSTMCells.push((data, c, h) =>
        tf.basicLSTMCell(
          selectionLSTMForgetBias,
          selectionModelVars[`${layerPrefix}/Matrix`],
          selectionModelVars[`${layerPrefix}/Bias`],
          data,
          c,
          h
        )
      );
    }
  }

  async function selectionDispose() {
    if (selectionModelVars === null) return;
    await dispose(selectionModelVars);
    selectionModelVars = null;
    selectionLSTMForgetBias.dispose();
  }

  function createInitialState(batchSize) {
    if (batchSize === undefined) {
      batchSize = 1;
    }
    const state = { c: [], h: [] };
    for (let i = 0; i < RNN_NUM_LAYERS; ++i) {
      state.c.push(tf.zeros([batchSize, RNN_CELL_SIZE], "float32"));
      state.h.push(tf.zeros([batchSize, RNN_CELL_SIZE], "float32"));
    }
    return state;
  }

  function disposeState(state) {
    for (let i = 0; i < state.c.length; ++i) {
      state.c[i].dispose();
      state.h[i].dispose();
    }
  }

  function stepToArray(step) {
    if (step < VOCAB_NUM_SPECIAL || step >= VOCAB_SIZE) {
      throw new Error("Invalid step");
    }
    const s = Number(step - 1)
      .toString(VOCAB_NUM_ARROW_TYPES)
      .padStart(VOCAB_NUM_ARROW_DIRECTIONS);
    const arr = [];
    for (let i = 0; i < VOCAB_NUM_ARROW_DIRECTIONS; ++i) {
      arr.push(Number(s[i]));
    }
    return arr;
  }

  function arrayToStep(arr) {
    let s = "";
    for (let i = 0; i < arr.length; ++i) {
      s += String(arr[i]);
    }
    const step = parseInt(s, VOCAB_NUM_ARROW_TYPES) + 1;
    if (step < VOCAB_NUM_SPECIAL || step >= VOCAB_SIZE) {
      throw new Error("Invalid array");
    }
    return step;
  }

  function lstmLogits(state, lastStep, dtPrev, dtNext) {
    if (selectionModelVars === null) {
      throw new Error("Must call initialize method first");
    }
    if (lastStep < 0 || lastStep > 256) {
      throw new Error("Invalid step");
    }

    const [finalState, logits] = tf.tidy(() => {
      // Step features
      const stepFeatsArr = new Float32Array(
        VOCAB_NUM_SPECIAL + VOCAB_NUM_ARROW_DIRECTIONS * VOCAB_NUM_ARROW_TYPES
      );
      for (let i = 0; i < stepFeatsArr.length; ++i) {
        stepFeatsArr[i] = 0;
      }
      if (lastStep === 0) {
        stepFeatsArr[0] = 1;
      } else {
        const lastStepArr = stepToArray(lastStep);
        for (let i = 0; i < VOCAB_NUM_ARROW_DIRECTIONS; ++i) {
          stepFeatsArr[
            VOCAB_NUM_SPECIAL + i * VOCAB_NUM_ARROW_TYPES + lastStepArr[i]
          ] = 1;
        }
      }
      const stepFeats = tf.tensor2d(
        stepFeatsArr,
        [1, stepFeatsArr.length],
        "float32"
      );

      // Time features
      const timeFeats = tf.tensor2d([dtPrev, dtNext], [1, 2], "float32");

      // Project feats array through RNN input matrix
      const a = tf.matMul(stepFeats, selectionModelVars["model_ss/rnn_proj/W"]);
      const b = tf.matMul(
        timeFeats,
        selectionModelVars["model_ss/rnn_proj/nosym_W"]
      );
      let x = tf.add(a, b);
      /*
      let x = tf.matMul(
        tf.concat([stepFeats, timeFeats], 1),
        tf.concat(
          [
            selectionModelVars["model_ss/rnn_proj/W"],
            selectionModelVars["model_ss/rnn_proj/nosym_W"]
          ],
          0
        )
      );
      */
      x = tf.add(x, selectionModelVars["model_ss/rnn_proj/b"]);

      // Evaluate RNN
      const [c, h] = tf.multiRNNCell(selectionLSTMCells, x, state.c, state.h);
      const finalState = { c: c, h: h };

      // Project to logits
      x = tf.matMul(
        h[RNN_NUM_LAYERS - 1],
        selectionModelVars["model_ss/sym_rnn_output/softmax_w"]
      );
      const logits = tf.add(
        x,
        selectionModelVars["model_ss/sym_rnn_output/softmax_b"]
      );

      return [finalState, logits];
    });

    return [finalState, logits];
  }

  async function select(timestamps, difficulty) {
    if (selectionModelVars === null) {
      throw new Error("Must call initialize method first");
    }

    const steps = tf.tidy(() => {
      let state = createInitialState();
      let lastStep = 0;
      for (let i = 0; i < timestamps.length; ++i) {
        const [finalState, logits] = lstmLogits(state, 0, 0, 0);
        const probs = tf.softmax(logits);
        disposeState(state);
        state = finalState;
      }
      disposeState(state);
    });

    return steps;
  }

  ddc.stepSelection = {};
  ddc.stepSelection.initialize = selectionInitialize;
  ddc.stepSelection.dispose = selectionDispose;
  ddc.stepSelection.select = select;

  ddc.choreograph = async audioUrl => {
    ddc.audioIO.loadFromUri();
  };
})(window.ddc, window.AudioContext || window.webkitAudioContext, window.tf);

/*
const model = new mm.OnsetsAndFrames(
  "https://storage.googleapis.com/magentadata/js/checkpoints/transcription/onsets_frames_uni"
);


async function transcribeFiles(fileList) {
  for (let i = 0; i < fileList.length; ++i) {
    const name = fileList[i].name;
    console.log("Transcribing " + name + " ...");
    const basename = name.split(".")[0];
    const outputFp = basename + ".mid";

    await model.transcribeFromAudioFile(fileList[i]).then(ns => {
      console.log("Done! " + ns.notes.length + " notes");
      saveAs(new File([mm.sequenceProtoToMidi(ns)], outputFp));
    });
  }
}
  const fileInput = document.getElementById("file-input");

  fileInput.addEventListener("change", function(e) {
    transcribeFiles(fileInput.files);
  });

*/
