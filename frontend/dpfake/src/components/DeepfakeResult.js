import React from "react";
import "./DeepfakeResult.css";
import { useLocation } from "react-router-dom";

export default function DeepfakeResult() {
  const location = useLocation();  
  const { prediction, probability, grad_cam_Video, output_box_Video,explanations, originalImage, heatmapImage } = location.state;
  

  return (

    <div className="page-container">
        {/* ✅ 상단 네비게이션 바 */}
        <nav className="navbar">
          <div className="navbar-logo" onClick={() => window.location.reload()}>DE-Fake it</div>
          <div className="navbar-menu">
            <a href="#about">About Us</a>
            <a href="#" onClick={() => window.location.reload()}>Home</a>
          </div>
        </nav>
      <h1 className="title">Real or Fake</h1>

        <div className="video-container" >
        {grad_cam_Video ? (
          <video controls
          className="video-player">
            <source src={output_box_Video} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ) : (
          <p>영상을 불러오는 중입니다...</p>
        )}
                {grad_cam_Video ? (
          <video controls
          className="video-player">
            <source src={grad_cam_Video} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ) : (
          <p>영상을 불러오는 중입니다...</p>
        )}
      </div>
    

      {/* <div className="video-container">
        <video
          src="/sample-video.mp4"
          controls
          className="video-player"
        >
          Your browser does not support the video tag.
        </video>
      </div> */}
      <h2 className="label-fake">{prediction}</h2>

      <h3 className="reason-title">Reason for Judgment</h3>
      <p className="reason-text">
        This video was classified as {prediction} with a {probability}% probability.<br />
        {explanations}
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
