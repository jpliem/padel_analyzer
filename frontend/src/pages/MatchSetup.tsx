import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createMatch } from '../api';

const MatchSetup: React.FC = () => {
  const navigate = useNavigate();
  const [name, setName] = useState('Match');
  const [format, setFormat] = useState('best_of_3');
  const [goldenPoint, setGoldenPoint] = useState(true);
  const [players, setPlayers] = useState({ P1: 'Player 1', P2: 'Player 2', P3: 'Player 3', P4: 'Player 4' });
  const [firstServer, setFirstServer] = useState('P1');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const result = await createMatch({
        match_name: name,
        players,
        teams: { TEAM_A: ['P1', 'P2'], TEAM_B: ['P3', 'P4'] },
        golden_point: goldenPoint,
        format,
      });
      navigate(`/match/${result.match_id}/calibrate`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const Toggle: React.FC<{ value: boolean; onToggle: (v: boolean) => void; labelTrue: string; labelFalse: string }> =
    ({ value, onToggle, labelTrue, labelFalse }) => (
      <div style={{ display: 'flex', gap: 4 }}>
        <button
          onClick={() => onToggle(true)}
          style={{
            flex: 1, padding: 8, textAlign: 'center', borderRadius: 6, fontSize: 13, fontWeight: 500, cursor: 'pointer',
            border: value ? '2px solid #1a1a2e' : '1px solid #d0d0d0',
            background: value ? '#1a1a2e' : 'white',
            color: value ? 'white' : '#555',
          }}
        >{labelTrue}</button>
        <button
          onClick={() => onToggle(false)}
          style={{
            flex: 1, padding: 8, textAlign: 'center', borderRadius: 6, fontSize: 13, fontWeight: 500, cursor: 'pointer',
            border: !value ? '2px solid #1a1a2e' : '1px solid #d0d0d0',
            background: !value ? '#1a1a2e' : 'white',
            color: !value ? 'white' : '#555',
          }}
        >{labelFalse}</button>
      </div>
    );

  return (
    <div style={{ padding: 32, maxWidth: 560, margin: '0 auto' }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>New Match</h1>
      {error && <div style={{ padding: 12, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 8, marginBottom: 16, color: '#e17055', fontSize: 13 }}>{error}</div>}
      <div style={{ marginBottom: 20 }}>
        <div className="label">Match Name</div>
        <input type="text" value={name} onChange={e => setName(e.target.value)} />
      </div>
      <div style={{ display: 'flex', gap: 16, marginBottom: 20 }}>
        <div style={{ flex: 1 }}>
          <div className="label">Format</div>
          <Toggle value={format === 'best_of_3'} onToggle={v => setFormat(v ? 'best_of_3' : 'best_of_1')} labelTrue="Best of 3" labelFalse="Best of 1" />
        </div>
        <div style={{ flex: 1 }}>
          <div className="label">Deuce Rule</div>
          <Toggle value={goldenPoint} onToggle={setGoldenPoint} labelTrue="Golden Point" labelFalse="Advantage" />
        </div>
      </div>
      <div style={{ display: 'flex', gap: 16, marginBottom: 20 }}>
        {[
          { label: 'Team A', color: '#74b9ff', ids: ['P1', 'P2'] },
          { label: 'Team B', color: '#e17055', ids: ['P3', 'P4'] },
        ].map(team => (
          <div key={team.label} style={{ flex: 1, background: 'white', border: '1px solid #e0e0e0', borderRadius: 10, padding: 16 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: team.color, textTransform: 'uppercase' as const, marginBottom: 12 }}>{team.label}</div>
            {team.ids.map((id, i) => (
              <div key={id} style={{ marginBottom: i === 0 ? 8 : 0 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{id}</div>
                <input type="text" value={players[id as keyof typeof players]} onChange={e => setPlayers({ ...players, [id]: e.target.value })} />
              </div>
            ))}
          </div>
        ))}
      </div>
      <div style={{ marginBottom: 24 }}>
        <div className="label">First Server</div>
        <div style={{ display: 'flex', gap: 4 }}>
          {['P1', 'P2', 'P3', 'P4'].map(id => (
            <button key={id} onClick={() => setFirstServer(id)}
              style={{
                flex: 1, padding: 8, textAlign: 'center', borderRadius: 6, fontSize: 13, cursor: 'pointer',
                border: firstServer === id ? '2px solid #1a1a2e' : '1px solid #d0d0d0',
                background: firstServer === id ? '#1a1a2e' : 'white',
                color: firstServer === id ? 'white' : '#555',
              }}
            >{id} — {players[id as keyof typeof players]}</button>
          ))}
        </div>
      </div>
      <button className="btn btn-primary" style={{ width: '100%', padding: 12, fontSize: 15 }} onClick={handleSubmit} disabled={submitting}>
        {submitting ? 'Creating...' : 'Create Match → Calibrate Court'}
      </button>
    </div>
  );
};

export default MatchSetup;
