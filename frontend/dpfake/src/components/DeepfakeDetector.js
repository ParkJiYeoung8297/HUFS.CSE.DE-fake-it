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
    const [analysisTableData, setAnalysisTableData] = useState([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false); // 🆕 업로드 이후 분석용 로딩


    const handleFileChange = async (e) => {
      const file = e.target.files[0];
      if (file && file.type.startsWith("video/")) {
        setVideoFile(file);

        console.log("Uploaded video:", file);
        setIsAnalyzing(false); 
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

        setIsAnalyzing(true); // ✅ 분석 시작

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
        setAnalysisTableData(result.table_data); // ⬅️ 테이블 데이터 저장
        
        navigate("/result", {
        state: {
          prediction:result.prediction,
          probability: result.probability,
          grad_cam_Video: result.grad_cam_video_url,
          output_box_Video: result.output_box_video_url,
          explanations: result.explanations,
          analysisTableHTML: result.table_data,
        },
      });

      } catch (error) {
        console.error("Upload failed:", error);
        alert("Video upload failed.");
      
      } finally {
    setIsAnalyzing(false); // ✅ 분석 끝
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
          {isUploading && <p>⏳ Processing video...</p>}

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

      {isAnalyzing && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p className="loading-text">⏳ Analyzing your video for deepfakes. Please wait...</p>
        </div>
      )}


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
          We developed this platform to reduce the social risks caused by deepfake content and to provide users reliable, explainable deepfake detection results. By analyzing facial information in videos, the system determines whether a video is REAL or FAKE, offering both visual explanations through Grad-CAM and natural language resoning via a Large Language MoDel(LLM). We focus on technical accurancy and interpretability, and our intuitive interface makes it easy for users to understand and utilize detection results. This platform is a reliable and accessible deepfake detection solution, aiming to become a tool for a safer digital environment.
        </p>
      </div>
    </div>
  </div>
  );
}
