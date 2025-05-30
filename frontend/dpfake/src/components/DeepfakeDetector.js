import React, { useState, useRef } from "react";
import "./DeepfakeDetector.css"; // 스타일 파일 import
import { useNavigate } from "react-router-dom";

export default function DeepfakeDetector() {

    const [videoFile, setVideoFile] = useState(null);
    const [convertedVideo, setConvertedVideo] = useState(null);
    const fileInputRef = useRef(null);
    const [isUploading, setIsUploading] = useState(false); // 🔄 대기 중 표시
    const navigate = useNavigate();

    const [probability, setProbability] = useState(null);
    // const [converted_video_url, setConvertedVideo] = useState(null);
    const [originalImage, setOriginalImage] = useState(null);
    const [heatmapImage, setHeatmapImage] = useState(null);


    const handleFileChange = async (e) => {
      const file = e.target.files[0];
      if (file && file.type.startsWith("video/")) {
        setVideoFile(file);

        console.log("Uploaded video:", file);

        setIsUploading(true); // 🔄 로딩 시작

        const formData = new FormData();
        formData.append("video", file);

        try {
          const response = await fetch("http://localhost:8000/showVideo/", {
            method: "POST",
            body: formData,
          });

          const result = await response.json();
          console.log("Server response:", result);

          setConvertedVideo(`http://localhost:8000${result.video_url}`);
        } catch (err) {
          console.error("Upload failed", err);
          alert("Upload failed.");
        } finally {
          setIsUploading(false); // ✅ 로딩 종료
        }



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

    const handleUploadToServer = async () => {
      if (!videoFile) {
        alert("Please select a video first.");
        return;
      }

      const formData = new FormData();
      formData.append("video", videoFile);

      try {
        const response = await fetch("http://localhost:8000/upload/", {
          method: "POST",
          body: formData,
        });

        const result = await response.json();
        console.log("Result from server:", result);

        setProbability(result.probability);
        // setConvertedVideo(result.converted_video_url);
        setOriginalImage(result.original_frame_url);
        setHeatmapImage(result.heatmap_url);
        
        navigate("/result", {
        state: {
          probability: result.probability,
          // convertedVideo: result.converted_video_url,
          originalImage: result.original_frame_url,
          heatmapImage: result.heatmap_url,
        },
      });

      } catch (error) {
        console.error("Upload failed:", error);
        alert("Video upload failed.");
      }
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
        <div className="upload-result" style={{ marginTop: "20px" }}>
          {isUploading && <p>⏳ 영상 처리 중입니다. 잠시만 기다려주세요...</p>}

          {!isUploading && convertedVideo && (
            <video controls width="100%" src={convertedVideo} />
          )}
        </div>


        {/* {convertedVideo && (
          <video controls width="100%" src={`http://localhost:8000${convertedVideo}`} />
        )} */}

        <div
          className="upload-dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={handleClickUpload}
        >
          <div className="upload-icon" style={{ display: videoFile ? "none" : "block" }}>🎥</div>
          <p style={{ display: videoFile ? "none" : "block" }}>
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
      </div>

      <button className="upload-btn" onClick={handleUploadToServer}>
        Start Deepfake detection
      </button>

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
