import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { autoDetectCourt, createMatch, startMatchAnalysis, uploadVideo } from '../api';

const MatchSetup: React.FC = () => {
  const navigate = useNavigate();
  const [video, setVideo] = useState<File | null>(null);
  const [name, setName] = useState('Friday padel');
  const [players, setPlayers] = useState({ P1: 'Player 1', P2: 'Player 2', P3: 'Player 3', P4: 'Player 4' });
  const [format, setFormat] = useState('best_of_3');
  const [goldenPoint, setGoldenPoint] = useState(true);
  const [firstServer, setFirstServer] = useState('P1');
  const [automaticCourt, setAutomaticCourt] = useState(true);
  const [step, setStep] = useState('');
  const [error, setError] = useState<string | null>(null);

  const videoLabel = useMemo(() => {
    if (!video) return 'MP4, MOV or phone recording';
    return `${video.name} · ${(video.size / 1024 / 1024).toFixed(1)} MB`;
  }, [video]);

  const submit = async () => {
    if (!video) {
      setError('Choose the match recording first.');
      return;
    }
    setError(null);
    try {
      setStep('Creating match…');
      const created = await createMatch({
        match_name: name.trim() || 'Padel match', players,
        teams: { TEAM_A: ['P1', 'P2'], TEAM_B: ['P3', 'P4'] },
        golden_point: goldenPoint, format, first_server: firstServer,
        out_of_court_play_enabled: false,
      });
      setStep('Uploading recording…');
      await uploadVideo(created.match_id, video, 'tracknet');

      if (!automaticCourt) {
        navigate(`/match/${created.match_id}/calibrate`);
        return;
      }

      setStep('Finding court lines…');
      try {
        await autoDetectCourt(created.match_id);
      } catch {
        navigate(`/match/${created.match_id}/calibrate`, {
          state: { notice: 'Automatic court detection needs a quick manual check.' },
        });
        return;
      }

      setStep('Starting smart analysis…');
      await startMatchAnalysis(created.match_id);
      navigate(`/match/${created.match_id}/analyze`);
    } catch (err: any) {
      setError(err.message || 'Could not create this match.');
      setStep('');
    }
  };

  return (
    <div className="setup-page">
      <section className="setup-intro">
        <span className="eyebrow">Single-camera smart recording</span>
        <h1>Turn one fixed-court video into a match you can review.</h1>
        <p>Get an annotated recording, rally moments, approximate court tracking and a queue for uncertain calls.</p>
        <div className="camera-guide">
          <strong>Best camera position</strong>
          <span>Centered behind one baseline, above head height, landscape, with the full court visible. Keep it fixed for the match.</span>
        </div>
        <div className="truth-note">One camera cannot guarantee depth or hidden-ball decisions. The app marks uncertain moments for a person to confirm.</div>
      </section>

      <section className="setup-card">
        <div className="setup-step"><span>1</span><div><strong>Add the recording</strong><small>The original stays available beside the annotated version.</small></div></div>
        <label className={`video-drop ${video ? 'has-file' : ''}`}>
          <input type="file" accept="video/mp4,video/quicktime,video/*" hidden
            onChange={event => setVideo(event.target.files?.[0] || null)} />
          <span className="video-drop-icon">{video ? '✓' : '↑'}</span>
          <strong>{video ? 'Recording ready' : 'Choose match video'}</strong>
          <small>{videoLabel}</small>
        </label>

        <div className="setup-step"><span>2</span><div><strong>Name the match and players</strong><small>You can use placeholders for a quick demo.</small></div></div>
        <label className="field-label">Match name<input value={name} onChange={event => setName(event.target.value)} /></label>
        <div className="team-grid">
          {(['P1', 'P2', 'P3', 'P4'] as const).map((id, index) => (
            <label className="field-label" key={id}>{index < 2 ? 'Team A' : 'Team B'} · {id}
              <input value={players[id]} onChange={event => setPlayers({ ...players, [id]: event.target.value })} />
            </label>
          ))}
        </div>

        <details className="advanced-settings">
          <summary>Match rules</summary>
          <div className="setting-row">
            <label>Format<select value={format} onChange={event => setFormat(event.target.value)}><option value="best_of_3">Best of 3</option><option value="best_of_1">One set</option></select></label>
            <label>Deuce<select value={goldenPoint ? 'golden' : 'advantage'} onChange={event => setGoldenPoint(event.target.value === 'golden')}><option value="golden">Golden point</option><option value="advantage">Advantage</option></select></label>
            <label>First server<select value={firstServer} onChange={event => setFirstServer(event.target.value)}>{Object.keys(players).map(id => <option key={id}>{id}</option>)}</select></label>
          </div>
        </details>

        <label className="auto-court"><input type="checkbox" checked={automaticCourt} onChange={event => setAutomaticCourt(event.target.checked)} /><span><strong>Find court lines automatically</strong><small>If it cannot, the app opens a manual court check.</small></span></label>
        {error && <div className="form-error">{error}</div>}
        <button className="btn btn-success create-analysis" onClick={submit} disabled={!!step}>{step || 'Create smart recording'}</button>
      </section>
    </div>
  );
};

export default MatchSetup;
