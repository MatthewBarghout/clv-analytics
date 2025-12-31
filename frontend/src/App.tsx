import React, { useEffect, useState } from 'react';
import Dashboard from './Dashboard';

function App() {
  const [apiStatus, setApiStatus] = useState<string>('Checking...');

  useEffect(() => {
    // Test API connection
    fetch('http://localhost:8000/api/health')
      .then(res => res.json())
      .then(data => setApiStatus('API Connected: ' + data.status))
      .catch(err => setApiStatus('API Error: ' + err.message));
  }, []);

  return (
    <div style={{ padding: '20px', backgroundColor: '#1a1a1a', minHeight: '100vh', color: 'white' }}>
      <h1 style={{ marginBottom: '20px' }}>CLV Analytics</h1>
      <p style={{ marginBottom: '20px' }}>Status: {apiStatus}</p>
      <Dashboard />
    </div>
  );
}

export default App;
