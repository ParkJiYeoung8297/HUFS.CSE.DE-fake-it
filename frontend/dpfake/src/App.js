import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    // 장고 API 호출
    fetch('/hello/') // 장고 API URL로 변경
      .then(response => response.json())
      .then(data => {
        // 데이터를 받아와서 상태 업데이트
        console.log('응답 데이터:', data); // 이거 추가!!
        setMessage(data.message);
      })
      .catch(error => console.error('Error fetching data: ', error));
  }, []);  // 빈 배열을 넣어 컴포넌트가 처음 마운트될 때만 실행되게 함

  return (
    <div className="App">
      <h1>Message from Django : {message}</h1>  {/* API에서 받은 메시지를 출력 */}
    </div>
  );
}

export default App;