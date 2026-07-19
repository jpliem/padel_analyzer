import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  cancelAnalysis, correctScore, deleteAnalysis, getAnalysisStatus, getAnnotatedVideoUrl, getExportUrl,
  getHighlightClipUrl, getMatchResult, getRecordingVideoUrl, resolveReview, startMatchAnalysis, uploadVideo,
} from '../api';
import Court3DView from '../components/Court3DView';
import CourtMiniMap from '../components/CourtMiniMap';
import EventLog from '../components/EventLog';
import ReviewQueue from '../components/ReviewQueue';
import Scoreboard from '../components/Scoreboard';
import type { AnalysisResult, EventData, Highlight, MatchData, ReviewRecord } from '../types';

type ViewState = 'loading' | 'needs_upload' | 'uploading' | 'processing' | 'complete' | 'error';

const formatTime = (seconds: number) => {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
};

const OfflineAnalysis: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const [state, setState] = useState<ViewState>('loading');
  const [percent, setPercent] = useState(0);
  const [error, setError] = useState('');
  const [match, setMatch] = useState<MatchData | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [videoMode, setVideoMode] = useState<'annotated' | 'original'>('annotated');
  const [panel, setPanel] = useState<'highlights' | 'events' | 'court'>('highlights');
  const [courtView, setCourtView] = useState<'2d' | '3d'>('2d');
  const [currentFrame, setCurrentFrame] = useState(0);

  const load = useCallback(async () => {
    if (!id) return null;
    const result = await getMatchResult(id);
    setMatch(result.match);
    setPercent(result.job.percent || 0);
    if (result.analysis) {
      setAnalysis(result.analysis);
      setState('complete');
    } else if (result.job.state === 'processing') {
      setState('processing');
    } else if (result.job.state === 'error') {
      setState('error');
      setError(result.job.error || 'Analysis failed.');
    } else if (result.match.media) {
      setState('loading');
    } else {
      setState('needs_upload');
    }
    return result;
  }, [id]);

  const poll = useCallback(() => {
    if (!id || pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await getAnalysisStatus(id);
        setPercent(status.percent || 0);
        if (status.state === 'complete') {
          clearInterval(pollRef.current);
          pollRef.current = undefined;
          await load();
        } else if (status.state === 'cancelled') {
          clearInterval(pollRef.current);
          pollRef.current = undefined;
          setState('error');
          setError('Analysis cancelled. You can retry whenever you are ready.');
        } else if (status.state === 'error') {
          clearInterval(pollRef.current);
          pollRef.current = undefined;
          setState('error');
          setError(status.error || 'Analysis failed.');
        }
      } catch (err: any) {
        setError(err.message);
      }
    }, 1000);
  }, [id, load]);

  useEffect(() => {
    (async () => {
      try {
        const result = await load();
        if (!result || result.analysis) return;
        if (result.job.state === 'processing') {
          poll();
        } else if (result.match.media && result.match.calibration) {
          await startMatchAnalysis(result.match.match_id);
          setState('processing');
          poll();
        }
      } catch (err: any) {
        setState('error');
        setError(err.message);
      }
    })();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [load, poll]);

  const upload = async (file: File) => {
    if (!id) return;
    try {
      setState('uploading');
      await uploadVideo(id, file);
      navigate(`/match/${id}/calibrate`);
    } catch (err: any) {
      setState('error');
      setError(err.message);
    }
  };

  const seek = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
    videoRef.current.play().catch(() => {});
  };

  const resolve = async (recordId: string, confirmed: boolean, winner?: number) => {
    if (!id) return;
    const response = await resolveReview(id, recordId, confirmed, winner);
    setAnalysis(previous => previous ? {
      ...previous,
      score: response.score,
      reviews: previous.reviews.map(item => item.id === recordId ? response.review : item),
    } : previous);
  };

  const addPoint = async (team: number) => {
    if (!id) return;
    const response = await correctScore(id, team);
    setAnalysis(previous => previous ? { ...previous, score: response.score } : previous);
    await load();
  };

  const reanalyze = async () => {
    if (!id || !window.confirm('Replace this analysis and run the recording again?')) return;
    await deleteAnalysis(id);
    setAnalysis(null);
    setPercent(0);
    await startMatchAnalysis(id);
    setState('processing');
    poll();
  };

  const retry = async () => {
    if (!id) return;
    try {
      setError('');
      setPercent(0);
      await startMatchAnalysis(id);
      setState('processing');
      poll();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const fps = analysis?.media?.fps || match?.media?.fps || 30;
  const trajectory = analysis?.trajectory || [];
  const currentBall = trajectory.reduce<any>((closest, point) =>
    !closest || Math.abs(point.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? point : closest, null);
  const ballTrail = trajectory
    .filter(point => point.frame <= currentFrame && point.frame > currentFrame - 3 * fps)
    .slice(-40)
    .map(point => ({ x: point.x, y: point.y, z: point.z || 0 }));
  const currentPositions = (analysis?.player_positions || []).reduce<any>((closest: any, item: any) =>
    !closest || Math.abs(item.frame - currentFrame) < Math.abs(closest.frame - currentFrame) ? item : closest, null);
  const players = (currentPositions?.players || []).map((player: any) => ({
    id: player.player_id || `#${player.track_id}`, x: player.x, y: player.y,
    label: player.player_id || `#${player.track_id}`,
    team: (player.player_id === 'P1' || player.player_id === 'P2' ? 'A' : 'B') as 'A' | 'B',
  }));

  if (state !== 'complete' || !analysis || !id) {
    return (
      <div className="analysis-state">
        <div className="analysis-state-card">
          <span className="eyebrow">{match?.match_name || 'Smart recording'}</span>
          {state === 'loading' && <><h1>Preparing your match…</h1><p>Checking the recording and court setup.</p></>}
          {state === 'uploading' && <><h1>Uploading recording…</h1><p>Keep this page open until the upload finishes.</p></>}
          {state === 'processing' && <>
            <h1>Analyzing the match</h1><p>Tracking the active ball, players, rallies and uncertain moments.</p>
            <div className="progress-track"><span style={{ width: `${percent}%` }} /></div><strong>{percent.toFixed(0)}%</strong>
            <div className="truth-note">Long recordings can take time. Progress is saved, and completed results survive a server restart.</div>
            <button className="btn btn-outline" onClick={() => id && cancelAnalysis(id).catch(err => setError(err.message))}>Cancel analysis</button>
          </>}
          {state === 'needs_upload' && <>
            <h1>Add a court recording</h1><p>Use a fixed landscape video with the complete court visible.</p>
            <label className="btn btn-success file-button">Choose video<input hidden type="file" accept="video/*" onChange={event => event.target.files?.[0] && upload(event.target.files[0])} /></label>
          </>}
          {state === 'error' && <><h1>Analysis stopped</h1><p className="form-error">{error}</p><div style={{ display: 'flex', justifyContent: 'center', gap: 8 }}><button className="btn btn-success" onClick={retry}>Retry analysis</button><button className="btn btn-outline" onClick={() => navigate(`/match/${id}/calibrate`)}>Check court setup</button></div></>}
        </div>
      </div>
    );
  }

  const reviews = analysis.reviews || [];
  const highlights = analysis.highlights || [];
  const events = analysis.events || [];
  const media = analysis.media || match?.media || {
    original_name: 'Legacy recording', fps: 30, frame_count: 0,
    duration_seconds: 0, width: 0, height: 0, size_bytes: 0, uploaded_at: '',
  };
  const stats = analysis.stats || {
    rallies: highlights.length, total_events: events.length, hits: 0, bounces: 0,
    wall_hits: 0, serves: 0, faults: 0, average_rally_seconds: 0,
    longest_rally_seconds: 0, ball_track_points: trajectory.length,
    frames_processed: 0, pending_reviews: 0,
  };
  const pending = reviews.filter(item => item.status === 'proposed').length;
  const model = analysis.model_info;
  const modelEvidence = model?.evidence;
  const activeBall = analysis.active_ball_diagnostics || {};
  const evidenceStatus = analysis.evidence_status;
  return (
    <div className="review-page">
      <header className="review-header">
        <div><button className="text-button" onClick={() => navigate('/')}>← Matches</button><h1>{match?.match_name}</h1><p>{media.original_name} · {formatTime(media.duration_seconds)} · {media.fps.toFixed(1)} fps</p></div>
        <div className="review-actions">
          <a className="btn btn-outline" href={getExportUrl(id, 'csv')}>Events CSV</a>
          <a className="btn btn-outline" href={getExportUrl(id, 'json')}>Full JSON</a>
          <button className="btn btn-outline" onClick={reanalyze}>Re-analyze</button>
        </div>
      </header>

      <div className="accuracy-banner"><strong>Single-camera estimate</strong><span>{analysis.accuracy_notice || 'Older analysis loaded with compatibility defaults. Re-analyze for the newest confidence and highlight fields.'}</span><em>{pending} moment{pending === 1 ? '' : 's'} waiting for review</em></div>

      {model && <section className="model-evidence-panel">
        <div><strong>Ball model: {model.id}</strong><span>{model.selection_reason || model.status}</span></div>
        {modelEvidence
          ? <div><strong>{(modelEvidence.recall * 100).toFixed(1)}% matched</strong><span>{modelEvidence.matched_labels}/{modelEvidence.visible_labels} visible labels on one held-out Panasonic rally, within {modelEvidence.tolerance_px}px. This is not club-wide or scoring accuracy.</span></div>
          : <div><strong>Not benchmarked</strong><span>This detector has no registered reviewed-label result.</span></div>}
        <div><strong>{activeBall.rejected_candidates || 0} candidates rejected</strong><span>{activeBall.uncertain_frames || 0} frames kept uncertain because the active ball was ambiguous or jumped implausibly.</span></div>
      </section>}
      {evidenceStatus && <div className="evidence-strip">
        <span><strong>Audio:</strong> {evidenceStatus.audio?.status || 'unknown'} ({evidenceStatus.audio_impulses || 0} impulses)</span>
        <span><strong>Pose:</strong> {evidenceStatus.pose === 'enabled' ? 'enabled' : 'not configured'}</span>
        <span><strong>Contact proposals:</strong> {evidenceStatus.contact_proposals || 0}</span>
        <span><strong>Rule decisions:</strong> {evidenceStatus.semantic_rule_decisions || 0}</span>
        <span>{evidenceStatus.scoring_policy}</span>
      </div>}
      {analysis.system_scope && <details className="scope-disclosure">
        <summary>What this build really uses</summary>
        <div>
          <section><strong>Running in this analysis</strong>{analysis.system_scope.runtime?.map(item => <span key={item}>✓ {item}</span>)}</section>
          <section><strong>Research demo only</strong>{analysis.system_scope.research_only?.map(item => <span key={item}>◌ {item}</span>)}</section>
          <section><strong>Not validated yet</strong>{analysis.system_scope.not_validated?.map(item => <span key={item}>! {item}</span>)}</section>
        </div>
      </details>}

      <div className="review-grid">
        <main className="video-column">
          <div className="video-toolbar">
            <div><button className={videoMode === 'annotated' ? 'active' : ''} onClick={() => setVideoMode('annotated')}>Smart overlay</button><button className={videoMode === 'original' ? 'active' : ''} onClick={() => setVideoMode('original')}>Original</button></div>
            <span>Frame {currentFrame.toLocaleString()}</span>
          </div>
          <div className="video-shell"><video ref={videoRef} key={videoMode} controls muted src={videoMode === 'annotated' ? getAnnotatedVideoUrl(id) : getRecordingVideoUrl(id)} onTimeUpdate={event => setCurrentFrame(Math.floor(event.currentTarget.currentTime * fps))} /></div>

          <div className="stats-grid">
            <Stat label="Rallies" value={stats.rallies} />
            <Stat label="Tracked hits" value={stats.hits} />
            <Stat label="Bounces" value={stats.bounces} />
            <Stat label="Wall contacts" value={stats.wall_hits} />
            <Stat label="Longest rally" value={`${stats.longest_rally_seconds}s`} />
          </div>

          <div className="score-review-card">
            <div><h2>Score</h2><p>Only confirmed point decisions are included.</p></div>
            <Scoreboard score={analysis.score} />
            <div className="manual-points"><span>Manual point</span><button onClick={() => addPoint(1)}>+ Team A</button><button onClick={() => addPoint(2)}>+ Team B</button></div>
          </div>
          <ReviewQueue reviews={reviews} onResolve={resolve} onSeek={frame => seek(frame / fps)} />
        </main>

        <aside className="moments-column">
          <nav className="panel-tabs"><button className={panel === 'highlights' ? 'active' : ''} onClick={() => setPanel('highlights')}>Rallies</button><button className={panel === 'events' ? 'active' : ''} onClick={() => setPanel('events')}>Events</button><button className={panel === 'court' ? 'active' : ''} onClick={() => setPanel('court')}>Court</button></nav>
          {panel === 'highlights' && <HighlightList matchId={id} highlights={highlights} onSeek={seek} />}
          {panel === 'events' && <EventLog events={events} onEventClick={(event: EventData) => seek(event.timestamp)} />}
          {panel === 'court' && <div className="court-panel">
            <div className="court-view-toggle">
              <button className={courtView === '2d' ? 'active' : ''} onClick={() => setCourtView('2d')}>2D</button>
              <button className={courtView === '3d' ? 'active' : ''} onClick={() => setCourtView('3d')}>3D</button>
              {courtView === '3d' && <small>Ball height from ballistic fit; flat when the fit was unreliable.</small>}
            </div>
            {courtView === '2d'
              ? <CourtMiniMap players={players} ballPosition={currentBall ? { x: currentBall.x, y: currentBall.y } : null} />
              : <div className="court-3d-shell"><Court3DView players={players} ballPosition={currentBall ? { x: currentBall.x, y: currentBall.y, z: currentBall.z || 0 } : null} ballTrail={ballTrail} /></div>}
          </div>}
        </aside>
      </div>
    </div>
  );
};

const Stat: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => <div className="stat"><strong>{value}</strong><span>{label}</span></div>;

const HighlightList: React.FC<{ matchId: string; highlights: Highlight[]; onSeek: (seconds: number) => void }> = ({ matchId, highlights, onSeek }) => (
  <div className="highlight-list">
    {highlights.length === 0 && <div className="empty-panel"><strong>No complete rallies detected</strong><span>Use the event list and review queue. The app will not invent highlights when evidence is missing.</span></div>}
    {highlights.map(item => <div key={item.id} className="highlight-card" role="button" tabIndex={0} onClick={() => onSeek(item.start_seconds)} onKeyDown={event => event.key === 'Enter' && onSeek(item.start_seconds)}>
      <span className="play-dot">▶</span><span><strong>{item.title}</strong><small>{formatTime(item.start_seconds)} – {formatTime(item.end_seconds)} · {item.end_reason.split('_').join(' ')}</small></span>
      <span className="highlight-tools">{item.needs_review && <em>Review</em>}<a href={getHighlightClipUrl(matchId, item.id)} onClick={event => event.stopPropagation()}>Download</a></span>
    </div>)}
  </div>
);

export default OfflineAnalysis;
