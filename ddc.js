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
    // TODO: Handle stereo data
    // TODO: Resample
    return buffer.getChannelData(0);
  }

  ddc.audioIO = {};
  ddc.audioIO.loadFromFile = loadFromFile;
  ddc.audioIO.loadFromUrl = loadFromUrl;
  ddc.audioIO.resample = null;

  /* Feature extraction module maps audio to spectrogram */

  const FFT_FRAME_LENGTHS = [1024, 2048, 4096];
  const FFT_FRAME_STEP = 512;

  let featureModelVars = null;

  async function featureInitialize(ckptDirUrl) {
    if (featureModelVars !== null) {
      await featureModelVars();
    }
    featureModelVars = await retrieveVars(ckptDirUrl);
  }

  async function featureDispose() {
    await dispose(featureModelVars);
    featureModelVars = null;
  }

  async function featureExtract(waveformArr) {
    if (featureModelVars === null) {
      throw new Error("Must call initialize method first");
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
          tf.add(melSpectrogram, tf.scalar(1e-16, "float32"))
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

      // Normalize features to have zero mean and unit variance
      feats = tf.sub(feats, featureModelVars["feats_mean"]);
      feats = tf.div(feats, featureModelVars["feats_std"]);

      return feats;
    });

    return feats;
  }

  ddc.featureExtraction = {};
  ddc.featureExtraction.initialize = featureInitialize;
  ddc.featureExtraction.dispose = featureDispose;
  ddc.featureExtraction.extract = featureExtract;

  /* Step placement module maps spectrogram to events */

  let placementModelVars = null;

  async function placementInitialize(ckptDirUrl) {
    if (placementModelVars !== null) {
      await placementDispose();
    }
    placementModelVars = await retrieveVars(ckptDirUrl);
  }

  async function placementDispose() {
    await dispose(placementModelVars);
    placementModelVars = null;
  }

  ddc.stepPlacement = {};
  ddc.stepPlacement.initialize = placementInitialize;
  ddc.stepPlacement.dispose = placementDispose;

  /* Step selection module maps timestamped events to choreography */

  /*
    this.decLSTMCells = [];
    this.decForgetBias = tf.scalar(1, 'float32');
    for (let i = 0; i < this.cfg.RNN_NLAYERS; ++i) {
      let cellFuseSpec: string;
      if (this.cfg.RNN_UNFUSED_LEGACY) {
        cellFuseSpec = 'basic_lstm_cell';
      } else {
        cellFuseSpec = 'lstm_cell';
      }

      let cellPrefix: string;
      if (this.cfg.RNN_SINGLELAYER_LEGACY) {
        cellPrefix = `phero_model/decoder/rnn/rnn/multi_rnn_cell/cell_0/${cellFuseSpec}/`;
      } else {
        cellPrefix = `phero_model/decoder/rnn/rnn/multi_rnn_cell/cell_${i}/${cellFuseSpec}/`;
      }

      this.decLSTMCells.push((data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(
          this.decForgetBias,
          this.modelVars[cellPrefix + 'kernel'] as tf.Tensor2D,
          this.modelVars[cellPrefix + 'bias'] as tf.Tensor1D,
          data, c, h
        ));
    }
    */

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
