import React from "react";
import "./DeepfakeResult.css";

export default function DeepfakeResult() {
  return (
    <div className="page-container">
      <h1 className="title">Real or Fake</h1>

      <div className="video-container">
        <video
          src="/sample-video.mp4"
          controls
          className="video-player"
        >
          Your browser does not support the video tag.
        </video>
      </div>

      <h2 className="label-fake">Fake</h2>
      <p className="probability">
        Probability: <span>89%</span>
      </p>

      <h3 className="reason-title">Reason for Judgment</h3>
      <p className="reason-text">
        Signs of deepfake content have been detected in the video. The Grad-CAM visualization of the face is shown below.
      </p>

      <div className="image-stack">
        <img
          src="/deepfake_mosaic_noise_eye.jpg"
          alt="Original Frame with Noise"
        />
        <img
          src="/deepfake_gradcam_eye_intense.jpg"
          alt="Grad-CAM Visualization"
        />
      </div>
    </div>
  );
}
