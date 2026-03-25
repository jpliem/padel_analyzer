import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { uploadVideo, startAnalysis, getAnalysisStatus, getScore, getEvents, getTrajectory, getAnnotatedVideoUrl, getPositions } from '../api';
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
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [percent, setPercent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [score, setScore] = useState<ScoreData | null>(null);
  const [events, setEvents] = useState<EventData[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);
  const [positions, setPositions] = useState<FramePositions[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [fps, setFps] = useState(24);
  const animRef = useRef<number>();

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
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [status, syncVideoTime]);

  // Get ball position at current frame
  const currentBall = trajectory.find(t => t.frame === currentFrame)
    || trajectory.reduce<TrajectoryPoint | null>((closest, t) =>
      !closest || Math.abs(t.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? t : closest, null);

  // Get ball trail (last 20 frames)
  const ballTrail = trajectory
    .filter(t => t.frame >= currentFrame - 20 && t.frame <= currentFrame)
    .map(t => ({ x: t.x, y: t.y }));

  // Get player positions at current frame
  const currentPositions = positions.find(p => p.frame === currentFrame)
    || positions.reduce<FramePositions | null>((closest, p) =>
      !closest || Math.abs(p.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? p : closest, null);

  const playerDots = (currentPositions?.players || []).map(p => ({
    id: p.player_id || `#${p.track_id}`,
    x: p.x,
    y: p.y,
    team: (p.player_id === 'P1' || p.player_id === 'P2' ? 'A' : 'B') as 'A' | 'B',
    label: p.player_id || `#${p.track_id}`,
  }));

  const handleUpload = async (file: File) => {
    if (!id) return;
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setStatus('uploading');
    setError(null);
    try {
      const { job_id } = await uploadVideo(id, file);
      setStatus('starting');
      await startAnalysis(job_id);
      setStatus('processing');
      const poll = setInterval(async () => {
        try {
          const s = await getAnalysisStatus(job_id);
          setPercent(s.percent);
          if (s.state === 'complete') {
            clearInterval(poll);
            setStatus('complete');
            const [scoreData, eventData, trajData, posData] = await Promise.all([
              getScore(id), getEvents(id), getTrajectory(id), getPositions(id),
            ]);
            setScore(scoreData);
            setEvents(eventData.events);
            setTrajectory(trajData.trajectory);
            setPositions(posData.positions);
          } else if (s.state === 'error') {
            clearInterval(poll);
            setStatus('error');
            setError(s.error || 'Analysis failed');
          }
        } catch {
          clearInterval(poll);
          setStatus('error');
          setError('Lost connection to backend');
        }
      }, 1000);
    } catch (err: any) {
      setStatus('error');
      setError(err.message);
    }
  };

  const seekToEvent = (event: EventData) => {
    if (videoRef.current) {
      videoRef.current.currentTime = event.timestamp;
      videoRef.current.play();
    }
  };

  // Load existing results on mount
  useEffect(() => {
    if (!id) return;
    Promise.all([getScore(id), getEvents(id), getTrajectory(id), getPositions(id)])
      .then(([s, e, t, p]) => {
        if (e.events.length > 0 || t.trajectory.length > 0) {
          setScore(s);
          setEvents(e.events);
          setTrajectory(t.trajectory);
          setPositions(p.positions);
          setStatus('complete');
        }
      })
      .catch(() => {});
  }, [id]);

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '12px 20px', background: 'white', borderBottom: '1px solid #e8e8e8' }}>
        <label className="btn btn-primary" style={{ cursor: 'pointer', fontSize: 12 }}>
          Upload Video
          <input type="file" accept="video/*" hidden onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} />
        </label>
        {videoFile && <span style={{ fontSize: 12, color: '#555' }}>{videoFile.name}</span>}
        {status === 'processing' && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ flex: 1, height: 6, background: '#e8e8e8', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${percent}%`, height: '100%', background: '#00b894', borderRadius: 3, transition: 'width 0.3s' }} />
            </div>
            <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500 }}>{percent.toFixed(0)}%</span>
          </div>
        )}
        {status === 'complete' && (
          <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500 }}>
            Complete — {trajectory.length} ball points, {events.length} events
          </span>
        )}
        {error && <span style={{ fontSize: 12, color: '#e17055' }}>{error}</span>}
      </div>

      {/* Main area */}
      <Group orientation="horizontal" style={{ flex: 1 }}>
        {/* Video panel */}
        <Panel defaultSize={55} minSize={30}>
          <div style={{ height: '100%', background: '#111', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {status === 'complete' && id ? (
                <video ref={videoRef} src={getAnnotatedVideoUrl(id)} controls muted
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
              ) : videoUrl ? (
                <video ref={videoRef} src={videoUrl} controls muted
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
              ) : (
                <div style={{ color: '#555', fontSize: 14 }}>Upload a video to begin analysis</div>
              )}
              {score && status !== 'complete' && (
                <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>
                  <Scoreboard score={score} variant="overlay" />
                </div>
              )}
            </div>
          </div>
        </Panel>

        <Separator style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />

        {/* Sidebar panel */}
        <Panel defaultSize={45} minSize={25}>
          <Group orientation="vertical" style={{ height: '100%' }}>
            {/* Score + Events (top) */}
            <Panel defaultSize={45} minSize={20}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <Scoreboard score={score} />
                <EventLog events={events} onEventClick={seekToEvent} />
              </div>
            </Panel>

            <Separator style={{ height: 4, background: '#e0e0e0', cursor: 'row-resize' }} />

            {/* 2D Court map (bottom) — now resizable! */}
            <Panel defaultSize={55} minSize={25}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CourtMiniMap
                  players={playerDots}
                  ballPosition={currentBall ? { x: currentBall.x, y: currentBall.y } : null}
                  ballTrail={ballTrail}
                />
                {status === 'complete' && (
                  <div style={{ padding: '6px 12px', background: '#1a2332', borderTop: '1px solid #2a3342', fontSize: 11, color: '#6b7b8d', flexShrink: 0 }}>
                    Frame {currentFrame} | Ball: {trajectory.length} pts | Players: {positions.length} frames | Events: {events.length}
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
