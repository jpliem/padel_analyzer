import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { listMatches } from '../api';
import MatchCard from '../components/MatchCard';
import type { MatchSummary } from '../types';

const Dashboard: React.FC = () => {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    listMatches()
      .then(setMatches)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ padding: 32, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700 }}>Matches</h1>
          <p style={{ fontSize: 13, color: '#888' }}>Your padel match analyses</p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/match/new')}>
          + New Match
        </button>
      </div>
      {error && (
        <div style={{ padding: 16, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 8, marginBottom: 16, color: '#e17055' }}>
          Backend not reachable: {error}
        </div>
      )}
      {loading ? (
        <p style={{ color: '#888' }}>Loading matches...</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
          {matches.map(m => (
            <MatchCard key={m.match_id} match={m}
              onDeleted={() => setMatches(prev => prev.filter(x => x.match_id !== m.match_id))} />
          ))}
          <div onClick={() => navigate('/match/new')}
            style={{
              background: 'white', border: '2px dashed #d0d0d0', borderRadius: 10,
              padding: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
              minHeight: 160, cursor: 'pointer', color: '#888',
            }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>+</div>
              <div style={{ fontSize: 14 }}>New Match</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
