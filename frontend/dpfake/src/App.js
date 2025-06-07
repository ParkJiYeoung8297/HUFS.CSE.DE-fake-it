import React, { useState, useEffect } from 'react';
import './App.css';

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DeepfakeDetector from "./components/DeepfakeDetector";
import DeepfakeResult from "./components/DeepfakeResult";

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/hello/') // 장고 API URL로 변경
    // fetch('/hello/')
      .then(response => response.json())
      .then(data => {
        setMessage(data.message);
      })
      .catch(error => console.error('Error fetching data: ', error));
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<DeepfakeDetector />} />
        <Route path="/result" element={<DeepfakeResult />} />
      </Routes>
    </Router>

  );

  // <DeepfakeDetector />;

}

export default App;