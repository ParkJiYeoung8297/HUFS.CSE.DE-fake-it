import React, { useState, useEffect } from 'react';
import DeepfakeDetector from "./components/DeepfakeDetector";
import './App.css';

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

  return <DeepfakeDetector />;

}

export default App;