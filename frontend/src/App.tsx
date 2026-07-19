import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/NavBar';
import Dashboard from './pages/Dashboard';
import MatchSetup from './pages/MatchSetup';
import Calibration from './pages/Calibration';
import OfflineAnalysis from './pages/OfflineAnalysis';
import LiveView from './pages/LiveView';

const App: React.FC = () => (
  <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
    <NavBar />
    <main style={{ flex: 1 }}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/match/new" element={<MatchSetup />} />
        <Route path="/match/:id/calibrate" element={<Calibration />} />
        <Route path="/match/:id/analyze" element={<OfflineAnalysis />} />
        <Route path="/match/:id/live" element={<LiveView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </main>
  </div>
);

export default App;
