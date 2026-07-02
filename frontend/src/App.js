import React from 'react';
import './App.css';

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DeepfakeDetector from "./components/DeepfakeDetector";
import DeepfakeResult from "./components/DeepfakeResult";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DeepfakeDetector />} />
        <Route path="/result" element={<DeepfakeResult />} />
      </Routes>
    </Router>

  );

}

export default App;
