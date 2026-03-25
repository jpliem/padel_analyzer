import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { startMatchAnalysis, getAnalysisStatus, getScore, getEvents, getTrajectory, getAnnotatedVideoUrl, getPositions, uploadVideo, startAnalysis, deleteAnalysis } from '../api';
import Scoreboard from '../components/Scoreboard';
import EventLog from '../components/EventLog';
import CourtMiniMap from '../components/CourtMiniMap';
import type { ScoreData, EventData, TrajectoryPoint } from '../types';

interface FramePositions {
  frame: number;
  players: { track_id: number; x: number; y: number; player_id: string | null }[];
}

const OfflineAnalysis: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<string>('loading');
  const [percent, setPercent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [score, setScore] = useState<ScoreData | null>(null);
  const [events, setEvents] = useState<EventData[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [positions, setPositions] = useState<FramePositions[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [fps] = useState(24);
  const animRef = useRef<number>();
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  const loadResults = useCallback(async () => {
    if (!id) return;
    const [sd, ed, td, pd] = await Promise.all([
      getScore(id), getEvents(id), getTrajectory(id), getPositions(id),
    ]);
    setScore(sd);
    setEvents(ed.events);
    setTrajectory(td.trajectory);
    setPositions(pd.positions);
    setStatus('complete');
  }, [id]);

  const startPolling = useCallback(() => {
    if (!id) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await getAnalysisStatus(id);
        setPercent(s.percent);
        if (s.state === 'complete') {
          clearInterval(pollRef.current);
          await loadResults();
        } else if (s.state === 'error') {
          clearInterval(pollRef.current);
          setStatus('error');
          setError(s.error || 'Analysis failed');
        }
      } catch {
        clearInterval(pollRef.current);
      }
    }, 1000);
  }, [id, loadResults]);

  // On mount: check for results → if none, auto-start analysis
  useEffect(() => {
    if (!id) return;

    (async () => {
      try {
        // Check if results already exist
        const [, e, t] = await Promise.all([
          getScore(id), getEvents(id), getTrajectory(id),
        ]);
        if (e.events.length > 0 || t.trajectory.length > 0) {
          await loadResults();
          return;
        }
      } catch {}

      // No results — try to start analysis (video should be uploaded from calibration)
      try {
        await startMatchAnalysis(id);
        setStatus('processing');
        startPolling();
      } catch (err: any) {
        // No video on server — show upload button
        setStatus('needs_upload');
      }
    })();

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [id, loadResults, startPolling]);

  // Manual upload (fallback if video wasn't uploaded during calibration)
  const handleManualUpload = async (file: File) => {
    if (!id) return;
    setStatus('uploading');
    setError(null);
    try {
      await uploadVideo(id, file);
      await startMatchAnalysis(id);
      setStatus('processing');
      startPolling();
    } catch (err: any) {
      setStatus('error');
      setError(err.message);
    }
  };

  // Sync video time to frame number
  const syncVideoTime = useCallback(() => {
    if (videoRef.current && status === 'complete') {
      const frame = Math.floor(videoRef.current.currentTime * fps);
      setCurrentFrame(frame);
    }
    animRef.current = requestAnimationFrame(syncVideoTime);
  }, [status, fps]);

  useEffect(() => {
    if (status === 'complete') {
      animRef.current = requestAnimationFrame(syncVideoTime);
    }
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [status, syncVideoTime]);

  const seekToEvent = (event: EventData) => {
    if (videoRef.current) {
      videoRef.current.currentTime = event.timestamp;
      videoRef.current.play();
    }
  };

  // Court map data synced to current frame
  const isOnCourt = (x: number, y: number) => x >= -2 && x <= 12 && y >= -3 && y <= 23;

  const rawBall = trajectory.reduce<TrajectoryPoint | null>((closest, t) =>
    !closest || Math.abs(t.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? t : closest, null);

  // Clamp ball to court bounds — hide if way off court
  const currentBall = rawBall && isOnCourt(rawBall.x, rawBall.y) ? rawBall : null;

  const ballTrail = trajectory
    .filter(t => t.frame >= currentFrame - 20 && t.frame <= currentFrame && isOnCourt(t.x, t.y))
    .map(t => ({ x: t.x, y: t.y }));

  const currentPositions = positions.reduce<FramePositions | null>((closest, p) =>
    !closest || Math.abs(p.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? p : closest, null);

  const playerDots = (currentPositions?.players || []).map(p => ({
    id: p.player_id || `#${p.track_id}`,
    x: p.x,
    y: p.y,
    team: (p.player_id === 'P1' || p.player_id === 'P2' ? 'A' : 'B') as 'A' | 'B',
    label: p.player_id || `#${p.track_id}`,
  }));

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '12px 20px', background: 'white', borderBottom: '1px solid #e8e8e8' }}>
        {status === 'needs_upload' && (
          <label className="btn btn-primary" style={{ cursor: 'pointer', fontSize: 12 }}>
            Upload Video
            <input type="file" accept="video/*" hidden onChange={e => e.target.files?.[0] && handleManualUpload(e.target.files[0])} />
          </label>
        )}
        {status === 'loading' && <span style={{ fontSize: 12, color: '#888' }}>Checking for results...</span>}
        {status === 'uploading' && <span style={{ fontSize: 12, color: '#888' }}>Uploading video...</span>}
        {(status === 'processing') && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: '#555', whiteSpace: 'nowrap' }}>Analyzing with TrackNet...</span>
            <div style={{ flex: 1, height: 6, background: '#e8e8e8', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${percent}%`, height: '100%', background: '#00b894', borderRadius: 3, transition: 'width 0.3s' }} />
            </div>
            <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500 }}>{percent.toFixed(0)}%</span>
          </div>
        )}
        {status === 'complete' && (
          <>
            <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500, flex: 1 }}>
              Complete — {trajectory.length} ball points, {events.length} events
            </span>
            <button
              className="btn btn-danger"
              style={{ fontSize: 11, padding: '4px 12px' }}
              onClick={async () => {
                if (!id || !window.confirm('Delete analysis results and re-analyze?')) return;
                await deleteAnalysis(id);
                setStatus('loading');
                setScore(null);
                setEvents([]);
                setTrajectory([]);
                setPositions([]);
                try {
                  await startMatchAnalysis(id);
                  setStatus('processing');
                  setPercent(0);
                  startPolling();
                } catch {
                  setStatus('needs_upload');
                }
              }}
            >
              Re-analyze
            </button>
          </>
        )}
        {error && <span style={{ fontSize: 12, color: '#e17055' }}>{error}</span>}
      </div>

      {/* Main area */}
      <Group orientation="horizontal" style={{ flex: 1 }}>
        <Panel defaultSize={55} minSize={30}>
          <div style={{ height: '100%', background: '#111', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {status === 'complete' && id ? (
                <video ref={videoRef} src={getAnnotatedVideoUrl(id)} controls muted
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
              ) : status === 'processing' ? (
                <div style={{ textAlign: 'center', color: '#888' }}>
                  <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
                  <div style={{ fontSize: 16 }}>Analyzing video...</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#00b894', marginTop: 8 }}>{percent.toFixed(0)}%</div>
                </div>
              ) : status === 'needs_upload' ? (
                <div style={{ color: '#888', fontSize: 14, textAlign: 'center' }}>
                  <div style={{ fontSize: 18, marginBottom: 8 }}>No video found</div>
                  <div>Upload a video using the button above</div>
                </div>
              ) : (
                <div style={{ color: '#555', fontSize: 14 }}>Loading...</div>
              )}
            </div>
          </div>
        </Panel>

        <Separator style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />

        <Panel defaultSize={45} minSize={25}>
          <Group orientation="vertical" style={{ height: '100%' }}>
            <Panel defaultSize={45} minSize={20}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <Scoreboard score={score} />
                <EventLog events={events} onEventClick={seekToEvent} />
              </div>
            </Panel>
            <Separator style={{ height: 4, background: '#e0e0e0', cursor: 'row-resize' }} />
            <Panel defaultSize={55} minSize={25}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CourtMiniMap
                  players={playerDots}
                  ballPosition={currentBall ? { x: currentBall.x, y: currentBall.y } : null}
                  ballTrail={ballTrail}
                />
                {status === 'complete' && (
                  <div style={{ padding: '6px 12px', background: '#1a2332', borderTop: '1px solid #2a3342', fontSize: 11, color: '#6b7b8d', flexShrink: 0 }}>
                    Frame {currentFrame} | Ball: {trajectory.length} pts | Events: {events.length}
                  </div>
                )}
              </div>
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  );
};

export default OfflineAnalysis;
