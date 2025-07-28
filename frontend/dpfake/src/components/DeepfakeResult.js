import React from "react";
import "./DeepfakeResult.css";
import { useLocation } from "react-router-dom";

export default function DeepfakeResult() {
  const location = useLocation();
  const {
    prediction,
    probability,
    grad_cam_Video,
    output_box_Video,
    explanations,
    analysisTableHTML,
  } = location.state;  

  return (
    <div className="page-container">
      {/* ✅ 네비게이션 바 */}
      <nav className="navbar">
        <div className="navbar-logo" onClick={() => window.location.reload()}>
          DE-Fake it
        </div>
        <div className="navbar-menu">
          <a href="#about">About Us</a>
          <a href="#" onClick={() => window.location.reload()}>
            Home
          </a>
        </div>
      </nav>

      <h1 className="title">DeepFake Analysis Report</h1>

      {/* ✅ 콘텐츠 영역 */}
      <div className="content-wrapper">
        {/* 🎞 왼쪽: 영상 + 결과 + 테이블 */}
        <div className="left-column">
          <div className="video-row">
            <video controls className="video-player">
              <source src={grad_cam_Video} type="video/mp4" />
            </video>
            <video controls className="video-player">
              <source src={output_box_Video} type="video/mp4" />
            </video>
          </div>
          <h2 className={prediction === "FAKE" ? "label-fake" : "label-real"}>
             {prediction} <span className={prediction === "FAKE" ? "probability-fake" : "probability-real"}>
              ({probability}%)
            </span> 
          </h2>
          <div
            // className="analysis-table"
            // dangerouslySetInnerHTML={{ __html: analysisTableHTML }}
          />
          <table className="analysis-table">
            <thead>
              <tr>
                <th>Facial Region</th>
                <th>First Count</th>
                <th>Second Count</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {analysisTableHTML.map((row, index) => (
                <tr key={index}>
                  <td>{row.region}</td>
                  <td>{row.first_count}</td>
                  <td>{row.second_count}</td>
                  <td>{row.region === "None" ? "-" : row.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
          

        </div>

        {/* 🧾 오른쪽: 설명 */}
        <div className="text-column">
          <h3 className="reason-title">Reason for Judgment</h3>
          <p className="reason-text">
            This video was classified as <strong>{prediction}</strong> with a{" "}
            <strong>{probability}%</strong> probability.
            <br />
            {explanations}
          </p>
        </div>
      </div>


    </div>
  );
}
