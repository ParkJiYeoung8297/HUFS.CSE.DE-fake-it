import React, { useState, useRef } from "react";
import "./DeepfakeDetector.css"; // 스타일 파일 import

export default function DeepfakeDetector() {

    const [videoFile, setVideoFile] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
      const file = e.target.files[0];
      if (file && file.type.startsWith("video/")) {
        setVideoFile(file);
        console.log("Uploaded video:", file);
      } else {
        alert("Please upload a valid video file.");
      }
    };

    const handleDrop = (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        setVideoFile(file);
      } else {
        alert("Only video files are allowed!");
      }
    };

    const handleDragOver = (e) => {
      e.preventDefault();
    };

    const handleClickUpload = () => {
      fileInputRef.current.click();
    };
  return (
    <div>
      {/* ✅ 상단 네비게이션 바 */}
      <nav className="navbar">
        <div className="navbar-logo" onClick={() => window.location.reload()}>DE-Fake it</div>
        <div className="navbar-menu">
          <a href="#about">About Us</a>
          <a href="#" onClick={() => window.location.reload()}>Home</a>
        </div>
      </nav>


    <div className="container">

      <h1 className="title">Detect deepfakes in videos</h1>
      <p className="subtitle">
        Upload a video to check for deepfakes and highlighted areas of concern.
      </p>

      <div className="upload-box">
        <div
          className="upload-dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={handleClickUpload}
        >
          <div className="upload-icon">🎥</div>
          <p>
            Drag & drop a video file here, or{" "}
            <span className="upload-click">click to upload</span>
          </p>
          <input
            type="file"
            accept="video/*"
            style={{ display: "none" }}
            ref={fileInputRef}
            onChange={handleFileChange}
          />
        </div>

        {videoFile && (
          <p className="uploaded-filename">Selected File: {videoFile.name}</p>
        )}
      </div>

      <h2 className="probability">Deepfake Probability: 89%</h2>

      <div className="images">
        <img
          src="/original.jpg"
          alt="Original Frame"
          className="image-box"
        />
        <img
          src="/heatmap.jpg"
          alt="Grad-CAM Heatmap"
          className="image-box"
        />
      </div>
      {/* ✅ About us 섹션 */}
      <div className="about-section" id="about" >
        <h2>About Us</h2>
        <p>
          We build AI tools that help people detect manipulated content and raise awareness about media trustworthiness.
        </p>
      </div>
    </div>
  </div>
  );
}
