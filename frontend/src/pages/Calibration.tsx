import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { calibrate } from '../api';
import CalibrationCanvas from '../components/CalibrationCanvas';
import CourtMiniMap from '../components/CourtMiniMap';

const Calibration: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [corners, setCorners] = useState<number[][]>([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCornerClick = (x: number, y: number) => {
    if (corners.length < 4) setCorners([...corners, [x, y]]);
  };

  const handleSave = async () => {
    if (!id || corners.length !== 4) return;
    setSaving(true);
    setError(null);
    try {
      await calibrate(id, corners);
      setSaved(true);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 56px)' }}>
      <div style={{ flex: 2 }}>
        <CalibrationCanvas videoFile={videoFile} corners={corners} onCornerClick={handleCornerClick} onReset={() => setCorners([])} />
      </div>
      <div style={{ flex: 1, minWidth: 280, padding: 16, display: 'flex', flexDirection: 'column', gap: 12, background: '#fafafa', borderLeft: '1px solid #e0e0e0' }}>
        <h2 style={{ fontSize: 16, fontWeight: 600 }}>Court Calibration</h2>
        <CourtMiniMap height={200} />
        <div>
          <div className="label">Video Source</div>
          <input type="file" accept="video/*" onChange={e => {
            const file = e.target.files?.[0];
            if (file) { setVideoFile(file); setCorners([]); setSaved(false); }
          }} style={{ fontSize: 13 }} />
        </div>
        {error && <div style={{ padding: 8, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 6, color: '#e17055', fontSize: 12 }}>{error}</div>}
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-outline" style={{ flex: 1, fontSize: 12 }} onClick={() => setCorners([])}>Reset Corners</button>
          <button className="btn btn-success" style={{ flex: 1, fontSize: 12 }} onClick={handleSave} disabled={corners.length !== 4 || saving}>
            {saving ? 'Saving...' : 'Save Calibration'}
          </button>
        </div>
        {saved && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8, padding: 12, background: '#f0fff4', border: '1px solid #00b894', borderRadius: 8 }}>
            <div style={{ fontSize: 13, color: '#00b894', fontWeight: 500 }}>Calibration saved!</div>
            <button className="btn btn-primary" onClick={() => navigate(`/match/${id}/analyze`)}>Analyze Video →</button>
            <button className="btn btn-outline" onClick={() => navigate(`/match/${id}/live`)}>Go Live →</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Calibration;
