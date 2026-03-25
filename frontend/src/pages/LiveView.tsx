import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { startLive, stopLive, correctScore } from '../api';
import { useWebSocket } from '../hooks/useWebSocket';
import CameraFeed from '../components/CameraFeed';
import Scoreboard from '../components/Scoreboard';
import EventLog from '../components/EventLog';
import CourtMiniMap from '../components/CourtMiniMap';

const LiveView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCorrect, setShowCorrect] = useState(false);

  const ws = useWebSocket(started ? 'ws://localhost:8000/live/stream' : null);

  useEffect(() => {
    if (!id) return;
    startLive({ match_id: id, device_id: 0, record: true })
      .then(() => setStarted(true))
      .catch(err => setError(err.message));
    return () => { stopLive().catch(() => {}); };
  }, [id]);

  const handleStop = async () => {
    try {
      await stopLive();
      navigate(`/match/${id}/analyze`);
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleCorrect = async (team: number) => {
    if (ws.connected) {
      ws.send({ type: 'correct', team });
    } else if (id) {
      await correctScore(id, team);
    }
    setShowCorrect(false);
  };

  if (error && !started) {
    return (
      <div style={{ padding: 32, textAlign: 'center' }}>
        <h2 style={{ color: '#e17055', marginBottom: 16 }}>Failed to start live session</h2>
        <p style={{ color: '#888', marginBottom: 16 }}>{error}</p>
        <button className="btn btn-primary" onClick={() => {
          setError(null);
          if (id) startLive({ match_id: id, device_id: 0, record: true }).then(() => setStarted(true)).catch(err => setError(err.message));
        }}>Retry</button>
      </div>
    );
  }

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 20px', background: 'white', borderBottom: '1px solid #e8e8e8' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: ws.connected ? '#e17055' : '#888' }} />
          <span style={{ fontSize: 12, color: ws.connected ? '#e17055' : '#888', fontWeight: 500 }}>
            {ws.connected ? 'LIVE' : 'Connecting...'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-outline" style={{ fontSize: 12, padding: '5px 12px' }} onClick={() => setShowCorrect(!showCorrect)}>
            Correct Score
          </button>
          <button className="btn btn-danger" style={{ fontSize: 12, padding: '5px 12px' }} onClick={handleStop}>
            Stop
          </button>
        </div>
      </div>
      {showCorrect && (
        <div style={{ padding: '8px 20px', background: '#fffbeb', borderBottom: '1px solid #fdcb6e', display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 13 }}>Award point to:</span>
          <button className="btn" style={{ background: '#74b9ff', color: 'white', fontSize: 12 }} onClick={() => handleCorrect(1)}>Team A</button>
          <button className="btn" style={{ background: '#e17055', color: 'white', fontSize: 12 }} onClick={() => handleCorrect(2)}>Team B</button>
          <button style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 12 }} onClick={() => setShowCorrect(false)}>Cancel</button>
        </div>
      )}
      <Group orientation="horizontal" style={{ flex: 1 }}>
        <Panel defaultSize={65} minSize={40}>
          <CameraFeed frameBase64={ws.lastFrame}>
            {ws.score && (
              <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>
                <Scoreboard score={ws.score} variant="overlay" />
              </div>
            )}
          </CameraFeed>
        </Panel>
        <Separator style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />
        <Panel defaultSize={35} minSize={20}>
          <Group orientation="vertical">
            <Panel defaultSize={50} minSize={20}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Scoreboard score={ws.score} />
                <EventLog events={ws.events} autoScroll />
              </div>
            </Panel>
            <Separator style={{ height: 4, background: '#e0e0e0', cursor: 'row-resize' }} />
            <Panel defaultSize={50} minSize={15}>
              <div style={{ height: '100%' }}>
                <CourtMiniMap height={300} />
              </div>
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  );
};

export default LiveView;
