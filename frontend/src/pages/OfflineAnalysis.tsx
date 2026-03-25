import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { uploadVideo, startAnalysis, getAnalysisStatus, getScore, getEvents, getTrajectory } from '../api';
import Scoreboard from '../components/Scoreboard';
import EventLog from '../components/EventLog';
import CourtMiniMap from '../components/CourtMiniMap';
import type { ScoreData, EventData, TrajectoryPoint } from '../types';

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
            const [scoreData, eventData, trajData] = await Promise.all([
              getScore(id), getEvents(id), getTrajectory(id),
            ]);
            setScore(scoreData);
            setEvents(eventData.events);
            setTrajectory(trajData.trajectory);
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

  // Load existing results on mount (when navigating back to an analyzed match)
  useEffect(() => {
    if (!id) return;
    Promise.all([getScore(id), getEvents(id), getTrajectory(id)])
      .then(([s, e, t]) => {
        // Show results if we have trajectory OR events (trajectory-only is valid)
        if (e.events.length > 0 || t.trajectory.length > 0) {
          setScore(s);
          setEvents(e.events);
          setTrajectory(t.trajectory);
          setStatus('complete');
        }
      })
      .catch(() => {});
  }, [id]);

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column' }}>
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
        {status === 'complete' && <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500 }}>Complete</span>}
        {error && <span style={{ fontSize: 12, color: '#e17055' }}>{error}</span>}
      </div>
      <Group orientation="horizontal" style={{ flex: 1 }}>
        <Panel defaultSize={65} minSize={40}>
          <div style={{ height: '100%', background: '#111', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
              {videoUrl ? (
                <video ref={videoRef} src={videoUrl} controls muted style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
              ) : (
                <div style={{ color: '#555', fontSize: 14 }}>Upload a video to begin analysis</div>
              )}
              {score && (
                <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>
                  <Scoreboard score={score} variant="overlay" />
                </div>
              )}
            </div>
          </div>
        </Panel>
        <Separator style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />
        <Panel defaultSize={35} minSize={20}>
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Scoreboard score={score} />
            <EventLog events={events} onEventClick={seekToEvent} />
            <CourtMiniMap
              height={140}
              ballTrail={trajectory.slice(-50).map(p => ({ x: p.x, y: p.y }))}
              ballPosition={trajectory.length > 0 ? { x: trajectory[trajectory.length - 1].x, y: trajectory[trajectory.length - 1].y } : null}
            />
            {status === 'complete' && (
              <div style={{ padding: '8px 12px', background: '#f8f8f8', borderTop: '1px solid #e8e8e8', fontSize: 12, color: '#888' }}>
                Trajectory: {trajectory.length} points | Events: {events.length}
                {events.length === 0 && trajectory.length > 0 && (
                  <div style={{ color: '#e17055', marginTop: 4 }}>
                    Ball detected but too sparse for event detection. TrackNetV2 needed for reliable scoring.
                  </div>
                )}
              </div>
            )}
          </div>
        </Panel>
      </Group>
    </div>
  );
};

export default OfflineAnalysis;
