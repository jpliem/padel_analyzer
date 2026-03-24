# Phase 3: Frontend Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the prototype frontend with a functional React app connected to all Phase 2 backend APIs — dashboard, match setup, calibration, offline analysis, and live view.

**Architecture:** React 18 + TypeScript + React Router for routing + react-resizable-panels for flexible layouts + Three.js for 3D court mini-map. Single `api.ts` client for all backend calls. Custom `useWebSocket` hook for live mode.

**Tech Stack:** React 18, TypeScript, React Router v6, react-resizable-panels, Three.js (@react-three/fiber), CRA (existing)

**Spec:** `docs/superpowers/specs/2026-03-24-phase3-frontend-rebuild-design.md`

**Working directory:** `/Users/jonathan/Documents/Github/padel_analyzer/frontend/` unless noted.

**Test command:** `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm test -- --watchAll=false`

---

## File Map

### New Files (frontend/src/)
| File | Responsibility |
|------|---------------|
| `types.ts` | TypeScript interfaces for all API data |
| `api.ts` | Backend API client — all fetch calls |
| `hooks/useWebSocket.ts` | WebSocket connection hook with auto-reconnect |
| `components/NavBar.tsx` | Top navigation bar |
| `components/Scoreboard.tsx` | Score display (overlay + sidebar variants) |
| `components/EventLog.tsx` | Scrollable event list with click-to-seek |
| `components/CourtMiniMap.tsx` | 3D court with Three.js (salvaged from prototype) |
| `components/MatchCard.tsx` | Dashboard match summary card |
| `components/CalibrationCanvas.tsx` | Clickable canvas for placing corner dots |
| `components/CameraFeed.tsx` | Canvas rendering WebSocket JPEG frames |
| `pages/Dashboard.tsx` | Match list landing page |
| `pages/MatchSetup.tsx` | New match form |
| `pages/Calibration.tsx` | Court corner calibration |
| `pages/OfflineAnalysis.tsx` | Upload, analyze, view results |
| `pages/LiveView.tsx` | Real-time camera + scoring |
| `styles/global.css` | Minimal global styles |

### Modified Files
| File | Changes |
|------|---------|
| `frontend/src/index.tsx` | Wrap App with BrowserRouter |
| `frontend/src/App.tsx` | Replace entirely — route definitions only |
| `frontend/package.json` | Add react-router-dom, react-resizable-panels |
| `backend/main.py` | Add `GET /matches` endpoint (~10 lines) |

### Deleted Files
| File | Reason |
|------|--------|
| `frontend/src/components/SetupDashboard.tsx` | Legacy prototype, replaced by Dashboard page |

---

### Task 1: Dependencies + Project Scaffolding

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/index.tsx`
- Rewrite: `frontend/src/App.tsx`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/styles/global.css`
- Delete: `frontend/src/components/SetupDashboard.tsx`

- [ ] **Step 1: Install new dependencies**

Run:
```bash
cd /Users/jonathan/Documents/Github/padel_analyzer/frontend
npm install react-router-dom react-resizable-panels
```

- [ ] **Step 2: Create TypeScript interfaces**

Create `src/types.ts`:

```typescript
export interface MatchSummary {
  match_id: string;
  match_name: string;
  status: string;
}

export interface MatchData {
  match_id: string;
  match_name: string;
  players: Record<string, string>;
  teams: Record<string, string[]>;
  golden_point: boolean;
  format: string;
  calibration: any | null;
}

export interface MatchSetupData {
  match_name: string;
  players: Record<string, string>;
  teams: Record<string, string[]>;
  golden_point: boolean;
  format: string;
}

export interface ScoreData {
  score: string;
  games: string;
  sets: string;
}

export interface EventData {
  event_type: string;
  timestamp: number;
  frame_number: number;
  position: { x: number; y: number };
  metadata: Record<string, any>;
}

export interface AnalysisStatus {
  state: string;
  percent: number;
  match_id?: string;
  error?: string;
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  z: number;
  speed: number;
  timestamp: number;
  frame: number;
  detected: boolean;
}

export interface LiveStartData {
  match_id: string;
  device_id: number;
  record: boolean;
}
```

- [ ] **Step 3: Create global CSS**

Create `src/styles/global.css`:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #fafafa;
  color: #1a1a2e;
}

a { text-decoration: none; color: inherit; }

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn:hover { opacity: 0.9; }
.btn-primary { background: #1a1a2e; color: white; }
.btn-success { background: #00b894; color: white; }
.btn-danger { background: #e17055; color: white; }
.btn-outline { background: white; border: 1px solid #1a1a2e; color: #1a1a2e; }

.badge {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
}
.badge-created { background: #e0e0e0; color: #555; }
.badge-calibrated { background: #fdcb6e; color: #333; }
.badge-analyzed { background: #00b894; color: white; }
.badge-live { background: #e17055; color: white; }

.label {
  font-size: 12px;
  font-weight: 600;
  color: #555;
  text-transform: uppercase;
  margin-bottom: 6px;
}

input[type="text"], input[type="number"] {
  padding: 10px 14px;
  border: 1px solid #d0d0d0;
  border-radius: 8px;
  font-size: 14px;
  width: 100%;
}
input[type="text"]:focus, input[type="number"]:focus {
  outline: none;
  border-color: #1a1a2e;
}
```

- [ ] **Step 4: Update index.tsx with BrowserRouter**

Rewrite `src/index.tsx`:

```typescript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles/global.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

- [ ] **Step 5: Rewrite App.tsx with routes (placeholder pages)**

Rewrite `src/App.tsx`:

```typescript
import React from 'react';
import { Routes, Route } from 'react-router-dom';
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
      </Routes>
    </main>
  </div>
);

export default App;
```

- [ ] **Step 6: Create placeholder pages (one-liner each)**

Create `src/pages/Dashboard.tsx`:
```typescript
import React from 'react';
const Dashboard: React.FC = () => <div style={{ padding: 32 }}>Dashboard — coming soon</div>;
export default Dashboard;
```

Create `src/pages/MatchSetup.tsx`:
```typescript
import React from 'react';
const MatchSetup: React.FC = () => <div style={{ padding: 32 }}>Match Setup — coming soon</div>;
export default MatchSetup;
```

Create `src/pages/Calibration.tsx`:
```typescript
import React from 'react';
const Calibration: React.FC = () => <div style={{ padding: 32 }}>Calibration — coming soon</div>;
export default Calibration;
```

Create `src/pages/OfflineAnalysis.tsx`:
```typescript
import React from 'react';
const OfflineAnalysis: React.FC = () => <div style={{ padding: 32 }}>Offline Analysis — coming soon</div>;
export default OfflineAnalysis;
```

Create `src/pages/LiveView.tsx`:
```typescript
import React from 'react';
const LiveView: React.FC = () => <div style={{ padding: 32 }}>Live View — coming soon</div>;
export default LiveView;
```

- [ ] **Step 7: Create NavBar component**

Create `src/components/NavBar.tsx`:

```typescript
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const NavBar: React.FC = () => {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <nav style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '12px 24px', background: '#fff', borderBottom: '1px solid #e0e0e0',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <Link to="/" style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e' }}>
          Padel Analyzer
        </Link>
        {!isHome && (
          <Link to="/" className="btn btn-outline" style={{ fontSize: 13, padding: '6px 14px' }}>
            Dashboard
          </Link>
        )}
      </div>
    </nav>
  );
};

export default NavBar;
```

- [ ] **Step 8: Delete legacy file**

```bash
rm frontend/src/components/SetupDashboard.tsx
```

- [ ] **Step 9: Verify the app builds and runs**

Run:
```bash
cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build
```
Expected: Build succeeds with no errors.

- [ ] **Step 10: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold frontend with React Router, 5 placeholder pages, NavBar, types, global CSS"
```

---

### Task 2: API Client + Backend GET /matches Endpoint

**Files:**
- Create: `frontend/src/api.ts`
- Modify: `backend/main.py` (add ~10 lines)
- Test: `backend/tests/test_api_phase2.py` (add 1 test)

- [ ] **Step 1: Create API client**

Create `frontend/src/api.ts`:

```typescript
import type {
  MatchSummary, MatchData, MatchSetupData, ScoreData,
  EventData, AnalysisStatus, TrajectoryPoint, LiveStartData,
} from './types';

const API = 'http://localhost:8000';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function listMatches(): Promise<MatchSummary[]> {
  const data = await fetchJSON<{ matches: MatchSummary[] }>(`${API}/matches`);
  return data.matches;
}

export async function createMatch(data: MatchSetupData): Promise<{ match_id: string }> {
  return fetchJSON(`${API}/match/setup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function getMatch(id: string): Promise<MatchData> {
  return fetchJSON(`${API}/match/${id}`);
}

export async function calibrate(id: string, corners: number[][]): Promise<void> {
  await fetchJSON(`${API}/match/${id}/calibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ corners }),
  });
}

export async function uploadVideo(matchId: string, file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  return fetchJSON(`${API}/analyze/upload?match_id=${matchId}`, {
    method: 'POST',
    body: form,
  });
}

export async function startAnalysis(jobId: string): Promise<void> {
  await fetchJSON(`${API}/analyze/start/${jobId}`, { method: 'POST' });
}

export async function getAnalysisStatus(jobId: string): Promise<AnalysisStatus> {
  return fetchJSON(`${API}/analyze/status/${jobId}`);
}

export async function getScore(id: string): Promise<ScoreData> {
  return fetchJSON(`${API}/match/${id}/score`);
}

export async function getEvents(id: string): Promise<{ events: EventData[] }> {
  return fetchJSON(`${API}/match/${id}/events`);
}

export async function getTrajectory(id: string): Promise<{ trajectory: TrajectoryPoint[] }> {
  return fetchJSON(`${API}/match/${id}/trajectory`);
}

export async function startLive(data: LiveStartData): Promise<void> {
  await fetchJSON(`${API}/live/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function stopLive(): Promise<void> {
  await fetchJSON(`${API}/live/stop`, { method: 'POST' });
}

export async function getReplay(): Promise<any> {
  return fetchJSON(`${API}/live/replay`);
}

export async function correctScore(matchId: string, team: number): Promise<void> {
  await fetchJSON(`${API}/match/${matchId}/correct-score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ team }),
  });
}
```

- [ ] **Step 2: Add GET /matches endpoint to backend**

Add to `backend/main.py` after the `root()` endpoint:

```python
@app.get("/matches")
def list_matches():
    if not os.path.exists(DATA_DIR):
        return {"matches": []}
    matches = []
    for match_id in os.listdir(DATA_DIR):
        config_path = os.path.join(DATA_DIR, match_id, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            status = "created"
            if config.get("calibration"):
                status = "calibrated"
            if match_id in _active_analyzers:
                status = "analyzed"
            matches.append({
                "match_id": match_id,
                "match_name": config.get("match_name", "Unknown"),
                "status": status,
            })
    return {"matches": matches}
```

- [ ] **Step 3: Add test for GET /matches**

Add to `backend/tests/test_api_phase2.py`:

```python
class TestListMatches:
    @pytest.mark.asyncio
    async def test_list_matches_empty(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/matches")
            assert resp.status_code == 200
            data = resp.json()
            assert "matches" in data
```

- [ ] **Step 4: Run backend tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer && source venv/bin/activate && cd backend && python -m pytest tests/ -v --tb=short`
Expected: All pass (132 + 1 new)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts backend/main.py backend/tests/test_api_phase2.py
git commit -m "feat: API client + GET /matches backend endpoint"
```

---

### Task 3: Dashboard Page + MatchCard Component

**Files:**
- Rewrite: `frontend/src/pages/Dashboard.tsx`
- Create: `frontend/src/components/MatchCard.tsx`

- [ ] **Step 1: Create MatchCard component**

Create `src/components/MatchCard.tsx`:

```typescript
import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { MatchSummary } from '../types';

interface Props {
  match: MatchSummary;
}

const statusBadge: Record<string, string> = {
  created: 'badge-created',
  calibrated: 'badge-calibrated',
  analyzed: 'badge-analyzed',
  live: 'badge-live',
};

const MatchCard: React.FC<Props> = ({ match }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (match.status === 'analyzed') {
      navigate(`/match/${match.match_id}/analyze`);
    } else if (match.status === 'calibrated') {
      // Show action buttons instead
      return;
    } else {
      navigate(`/match/${match.match_id}/calibrate`);
    }
  };

  return (
    <div
      onClick={handleClick}
      style={{
        background: 'white', border: '1px solid #e8e8e8', borderRadius: 10,
        padding: 20, cursor: 'pointer', transition: 'box-shadow 0.2s',
      }}
      onMouseOver={e => (e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)')}
      onMouseOut={e => (e.currentTarget.style.boxShadow = 'none')}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{match.match_name}</div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{match.match_id}</div>
        </div>
        <span className={`badge ${statusBadge[match.status] || 'badge-created'}`}>
          {match.status}
        </span>
      </div>

      {match.status === 'calibrated' && (
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <button
            className="btn btn-primary"
            style={{ fontSize: 12 }}
            onClick={e => { e.stopPropagation(); navigate(`/match/${match.match_id}/analyze`); }}
          >
            Analyze Video
          </button>
          <button
            className="btn btn-outline"
            style={{ fontSize: 12 }}
            onClick={e => { e.stopPropagation(); navigate(`/match/${match.match_id}/live`); }}
          >
            Go Live
          </button>
        </div>
      )}
    </div>
  );
};

export default MatchCard;
```

- [ ] **Step 2: Implement Dashboard page**

Rewrite `src/pages/Dashboard.tsx`:

```typescript
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
          {matches.map(m => <MatchCard key={m.match_id} match={m} />)}
          <div
            onClick={() => navigate('/match/new')}
            style={{
              background: 'white', border: '2px dashed #d0d0d0', borderRadius: 10,
              padding: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
              minHeight: 160, cursor: 'pointer', color: '#888',
            }}
          >
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
```

- [ ] **Step 3: Verify build**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add frontend/src/
git commit -m "feat: Dashboard page with match cards and API integration"
```

---

### Task 4: Match Setup Page

**Files:**
- Rewrite: `frontend/src/pages/MatchSetup.tsx`

- [ ] **Step 1: Implement Match Setup form**

Rewrite `src/pages/MatchSetup.tsx`:

```typescript
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
            <div style={{ fontSize: 12, fontWeight: 600, color: team.color, textTransform: 'uppercase', marginBottom: 12 }}>{team.label}</div>
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
            <button
              key={id}
              onClick={() => setFirstServer(id)}
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
```

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/MatchSetup.tsx
git commit -m "feat: Match Setup page with form, toggles, team cards, first server selector"
```

---

### Task 5: CourtMiniMap Component (salvaged from prototype)

**Files:**
- Create: `frontend/src/components/CourtMiniMap.tsx`

- [ ] **Step 1: Create CourtMiniMap**

Salvage the `PadelCourt3D` from the old App.tsx and add player/ball dot support.

Create `src/components/CourtMiniMap.tsx`:

```typescript
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Plane, Grid, Box } from '@react-three/drei';
import * as THREE from 'three';

interface PlayerDot {
  id: string;
  x: number;
  y: number;
  team: 'A' | 'B';
}

interface Props {
  players?: PlayerDot[];
  ballPosition?: { x: number; y: number } | null;
  ballTrail?: { x: number; y: number }[];
  height?: number;
}

const Court: React.FC = () => (
  <group>
    {/* Court surface */}
    <Plane args={[10, 20]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <meshStandardMaterial color="#1b5e20" />
    </Plane>
    <Grid args={[10, 20]} position={[0, 0.01, 0]} sectionColor="white" cellColor="#2e7d32" />
    {/* Net */}
    <Box args={[10, 0.88, 0.05]} position={[0, 0.44, 0]}>
      <meshStandardMaterial color="white" transparent opacity={0.6} />
    </Box>
    {/* Service lines */}
    <Box args={[10, 0.02, 0.02]} position={[0, 0.01, -3.05]}>
      <meshStandardMaterial color="white" />
    </Box>
    <Box args={[10, 0.02, 0.02]} position={[0, 0.01, 3.05]}>
      <meshStandardMaterial color="white" />
    </Box>
    <Box args={[0.02, 0.02, 6.1]} position={[0, 0.01, 0]}>
      <meshStandardMaterial color="white" />
    </Box>
  </group>
);

const PlayerMarker: React.FC<{ position: [number, number, number]; color: string; label: string }> = ({ position, color, label }) => (
  <mesh position={position}>
    <sphereGeometry args={[0.2]} />
    <meshStandardMaterial color={color} />
  </mesh>
);

const BallMarker: React.FC<{ position: [number, number, number] }> = ({ position }) => (
  <mesh position={position}>
    <sphereGeometry args={[0.12]} />
    <meshStandardMaterial color="#fdcb6e" emissive="#fdcb6e" emissiveIntensity={0.5} />
  </mesh>
);

const CourtMiniMap: React.FC<Props> = ({ players = [], ballPosition, ballTrail = [], height = 200 }) => (
  <div style={{ height, background: '#1a1a2e', borderRadius: 8, overflow: 'hidden' }}>
    <Canvas camera={{ position: [0, 18, 12], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.4} />
      <Court />
      {players.map(p => (
        <PlayerMarker
          key={p.id}
          position={[p.x - 5, 0.2, p.y - 10]}
          color={p.team === 'A' ? '#74b9ff' : '#e17055'}
          label={p.id}
        />
      ))}
      {ballPosition && (
        <BallMarker position={[ballPosition.x - 5, 0.3, ballPosition.y - 10]} />
      )}
      {ballTrail.map((p, i) => (
        <mesh key={i} position={[p.x - 5, 0.15, p.y - 10]}>
          <sphereGeometry args={[0.08]} />
          <meshStandardMaterial color="#fdcb6e" transparent opacity={0.2 + (i / ballTrail.length) * 0.6} />
        </mesh>
      ))}
      <OrbitControls enablePan={false} enableZoom={false} />
    </Canvas>
  </div>
);

export default CourtMiniMap;
```

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/CourtMiniMap.tsx
git commit -m "feat: CourtMiniMap 3D component with player dots and ball trail"
```

---

### Task 6: Shared Components — Scoreboard + EventLog

**Files:**
- Create: `frontend/src/components/Scoreboard.tsx`
- Create: `frontend/src/components/EventLog.tsx`

- [ ] **Step 1: Create Scoreboard component**

Create `src/components/Scoreboard.tsx`:

```typescript
import React from 'react';
import type { ScoreData } from '../types';

interface Props {
  score: ScoreData | null;
  teamA?: string;
  teamB?: string;
  variant?: 'overlay' | 'sidebar';
}

const Scoreboard: React.FC<Props> = ({ score, teamA = 'Team A', teamB = 'Team B', variant = 'sidebar' }) => {
  if (!score) return null;

  if (variant === 'overlay') {
    return (
      <div style={{
        display: 'flex', background: 'rgba(0,0,0,0.85)', borderRadius: 8, overflow: 'hidden', padding: 2,
      }}>
        <div style={{ padding: '8px 20px', textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: '#74b9ff', textTransform: 'uppercase' }}>{teamA}</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'white' }}>{score.score.split(' - ')[0]}</div>
        </div>
        <div style={{ width: 1, background: '#444' }} />
        <div style={{ padding: '8px 20px', textAlign: 'center' }}>
          <div style={{ fontSize: 9, color: '#e17055', textTransform: 'uppercase' }}>{teamB}</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'white' }}>{score.score.split(' - ')[1]}</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '12px 16px', background: 'white', borderBottom: '1px solid #e8e8e8' }}>
      <div className="label">Score</div>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 16, alignItems: 'baseline', marginTop: 8 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#74b9ff' }}>{teamA}</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{score.score.split(' - ')[0]}</div>
        </div>
        <div style={{ fontSize: 14, color: '#888' }}>-</div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#e17055' }}>{teamB}</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{score.score.split(' - ')[1]}</div>
        </div>
      </div>
      <div style={{ textAlign: 'center', fontSize: 12, color: '#888', marginTop: 4 }}>
        Games: {score.games} | Sets: {score.sets}
      </div>
    </div>
  );
};

export default Scoreboard;
```

- [ ] **Step 2: Create EventLog component**

Create `src/components/EventLog.tsx`:

```typescript
import React from 'react';
import type { EventData } from '../types';

interface Props {
  events: EventData[];
  onEventClick?: (event: EventData) => void;
  autoScroll?: boolean;
}

const eventColors: Record<string, string> = {
  BOUNCE: '#00b894',
  SERVE: '#6c5ce7',
  FAULT: '#e17055',
  HIT: '#6c5ce7',
  POINT_END: '#fdcb6e',
  LET: '#74b9ff',
};

const formatTime = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};

const EventLog: React.FC<Props> = ({ events, onEventClick, autoScroll = false }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [events.length, autoScroll]);

  const sorted = autoScroll ? [...events].reverse() : events;

  return (
    <div ref={containerRef} style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }}>
      <div className="label" style={{ marginBottom: 8 }}>Events</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {sorted.map((event, i) => {
          const isPoint = event.event_type === 'POINT_END';
          return (
            <div
              key={i}
              onClick={() => onEventClick?.(event)}
              style={{
                display: 'flex', gap: 8, padding: '6px 8px', borderRadius: 6, fontSize: 12,
                background: isPoint ? '#fffbeb' : 'white',
                border: isPoint ? '1px solid #fdcb6e' : '1px solid #e8e8e8',
                cursor: onEventClick ? 'pointer' : 'default',
              }}
            >
              <div style={{ color: '#888', minWidth: 32 }}>{formatTime(event.timestamp)}</div>
              <div style={{ color: eventColors[event.event_type] || '#888', minWidth: 16 }}>
                {isPoint ? '★' : '●'}
              </div>
              <div style={{ color: '#333' }}>
                {event.event_type.replace('_', ' ')}
                {event.metadata?.side ? ` (${event.metadata.side})` : ''}
                {event.metadata?.reason ? ` — ${event.metadata.reason}` : ''}
              </div>
            </div>
          );
        })}
        {events.length === 0 && (
          <div style={{ color: '#888', fontSize: 13, padding: 8 }}>No events yet</div>
        )}
      </div>
    </div>
  );
};

export default EventLog;
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/Scoreboard.tsx frontend/src/components/EventLog.tsx
git commit -m "feat: Scoreboard (overlay + sidebar) and EventLog components"
```

---

### Task 7: Calibration Page + CalibrationCanvas

**Files:**
- Create: `frontend/src/components/CalibrationCanvas.tsx`
- Rewrite: `frontend/src/pages/Calibration.tsx`

- [ ] **Step 1: Create CalibrationCanvas component**

Create `src/components/CalibrationCanvas.tsx`:

```typescript
import React, { useRef, useState, useEffect } from 'react';

interface Props {
  videoFile: File | null;
  corners: number[][];
  onCornerClick: (x: number, y: number) => void;
  onReset: () => void;
}

const CalibrationCanvas: React.FC<Props> = ({ videoFile, corners, onCornerClick, onReset }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoDims, setVideoDims] = useState({ w: 1280, h: 720 });

  useEffect(() => {
    if (!videoFile || !videoRef.current) return;
    const video = videoRef.current;
    video.src = URL.createObjectURL(videoFile);
    video.onloadedmetadata = () => {
      setVideoDims({ w: video.videoWidth, h: video.videoHeight });
      video.currentTime = 0.1; // grab first frame
    };
    video.onseeked = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.drawImage(video, 0, 0);
    };
  }, [videoFile]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !videoFile) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Redraw frame
    if (videoRef.current) ctx.drawImage(videoRef.current, 0, 0);

    // Draw corner dots and lines
    if (corners.length > 0) {
      ctx.strokeStyle = corners.length === 4 ? '#00b894' : '#74b9ff';
      ctx.lineWidth = 2;
      ctx.setLineDash(corners.length === 4 ? [] : [6, 4]);

      ctx.beginPath();
      corners.forEach((c, i) => {
        if (i === 0) ctx.moveTo(c[0], c[1]);
        else ctx.lineTo(c[0], c[1]);
      });
      if (corners.length === 4) ctx.closePath();
      ctx.stroke();

      corners.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c[0], c[1], 8, 0, Math.PI * 2);
        ctx.fillStyle = '#74b9ff';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'white';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), c[0], c[1]);
      });
    }
  }, [corners, videoFile]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (corners.length >= 4) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = videoDims.w / rect.width;
    const scaleY = videoDims.h / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    onCornerClick(Math.round(x), Math.round(y));
  };

  return (
    <div style={{ position: 'relative', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
      <video ref={videoRef} style={{ display: 'none' }} muted />
      {videoFile ? (
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          style={{ maxWidth: '100%', maxHeight: '100%', cursor: corners.length < 4 ? 'crosshair' : 'default' }}
        />
      ) : (
        <div style={{ color: '#888', textAlign: 'center', padding: 40 }}>
          <div style={{ fontSize: 18, marginBottom: 8 }}>Upload a video to calibrate</div>
          <div style={{ fontSize: 13 }}>First frame will be shown for corner selection</div>
        </div>
      )}

      {videoFile && (
        <div style={{ position: 'absolute', top: 12, left: 12, padding: '6px 12px', background: 'rgba(0,0,0,0.7)', borderRadius: 6, color: 'white', fontSize: 12 }}>
          Click 4 court corners: near-left → near-right → far-right → far-left
        </div>
      )}

      <div style={{ position: 'absolute', bottom: 12, left: 12, padding: '4px 10px', background: 'rgba(116,185,255,0.2)', border: '1px solid #74b9ff', borderRadius: 6, color: '#74b9ff', fontSize: 11 }}>
        {corners.length}/4 corners set
      </div>
    </div>
  );
};

export default CalibrationCanvas;
```

- [ ] **Step 2: Implement Calibration page**

Rewrite `src/pages/Calibration.tsx`:

```typescript
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
    if (corners.length < 4) {
      setCorners([...corners, [x, y]]);
    }
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
      {/* Left: Video frame */}
      <div style={{ flex: 2 }}>
        <CalibrationCanvas
          videoFile={videoFile}
          corners={corners}
          onCornerClick={handleCornerClick}
          onReset={() => setCorners([])}
        />
      </div>

      {/* Right: Controls */}
      <div style={{ flex: 1, minWidth: 280, padding: 16, display: 'flex', flexDirection: 'column', gap: 12, background: '#fafafa', borderLeft: '1px solid #e0e0e0' }}>
        <h2 style={{ fontSize: 16, fontWeight: 600 }}>Court Calibration</h2>

        {/* 3D Preview */}
        <CourtMiniMap height={200} />

        {/* Video source */}
        <div>
          <div className="label">Video Source</div>
          <input
            type="file"
            accept="video/*"
            onChange={e => {
              const file = e.target.files?.[0];
              if (file) { setVideoFile(file); setCorners([]); setSaved(false); }
            }}
            style={{ fontSize: 13 }}
          />
        </div>

        {error && <div style={{ padding: 8, background: '#fff3f3', border: '1px solid #e17055', borderRadius: 6, color: '#e17055', fontSize: 12 }}>{error}</div>}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-outline" style={{ flex: 1, fontSize: 12 }} onClick={() => setCorners([])}>
            Reset Corners
          </button>
          <button
            className="btn btn-success"
            style={{ flex: 1, fontSize: 12 }}
            onClick={handleSave}
            disabled={corners.length !== 4 || saving}
          >
            {saving ? 'Saving...' : 'Save Calibration'}
          </button>
        </div>

        {/* Post-save navigation */}
        {saved && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8, padding: 12, background: '#f0fff4', border: '1px solid #00b894', borderRadius: 8 }}>
            <div style={{ fontSize: 13, color: '#00b894', fontWeight: 500 }}>Calibration saved!</div>
            <button className="btn btn-primary" onClick={() => navigate(`/match/${id}/analyze`)}>
              Analyze Video →
            </button>
            <button className="btn btn-outline" onClick={() => navigate(`/match/${id}/live`)}>
              Go Live →
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Calibration;
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/CalibrationCanvas.tsx frontend/src/pages/Calibration.tsx
git commit -m "feat: Calibration page with interactive corner clicking and 3D preview"
```

---

### Task 8: Offline Analysis Page

**Files:**
- Rewrite: `frontend/src/pages/OfflineAnalysis.tsx`

- [ ] **Step 1: Implement Offline Analysis page**

Rewrite `src/pages/OfflineAnalysis.tsx`:

```typescript
import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
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

      // Poll
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

  // Try loading existing results on mount
  useEffect(() => {
    if (!id) return;
    Promise.all([getScore(id), getEvents(id), getTrajectory(id)])
      .then(([s, e, t]) => {
        if (e.events.length > 0) {
          setScore(s);
          setEvents(e.events);
          setTrajectory(t.trajectory);
          setStatus('complete');
        }
      })
      .catch(() => {}); // No results yet, that's fine
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
        {status === 'complete' && <span style={{ fontSize: 12, color: '#00b894', fontWeight: 500 }}>Complete</span>}
        {error && <span style={{ fontSize: 12, color: '#e17055' }}>{error}</span>}
      </div>

      {/* Main area */}
      <PanelGroup direction="horizontal" style={{ flex: 1 }}>
        {/* Video panel */}
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

        <PanelResizeHandle style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />

        {/* Sidebar panel */}
        <Panel defaultSize={35} minSize={20}>
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Scoreboard score={score} />
            <EventLog events={events} onEventClick={seekToEvent} />
            <CourtMiniMap height={140} />
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
};

export default OfflineAnalysis;
```

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/OfflineAnalysis.tsx
git commit -m "feat: Offline Analysis page with upload, progress, resizable panels, results"
```

---

### Task 9: WebSocket Hook + CameraFeed Component

**Files:**
- Create: `frontend/src/hooks/useWebSocket.ts`
- Create: `frontend/src/components/CameraFeed.tsx`

- [ ] **Step 1: Create useWebSocket hook**

Create `src/hooks/useWebSocket.ts`:

```typescript
import { useState, useEffect, useRef, useCallback } from 'react';
import type { ScoreData, EventData } from '../types';

interface WebSocketState {
  connected: boolean;
  lastFrame: string | null;
  score: ScoreData | null;
  events: EventData[];
  send: (data: any) => void;
}

export function useWebSocket(url: string | null): WebSocketState {
  const [connected, setConnected] = useState(false);
  const [lastFrame, setLastFrame] = useState<string | null>(null);
  const [score, setScore] = useState<ScoreData | null>(null);
  const [events, setEvents] = useState<EventData[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (!url) return;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        switch (msg.type) {
          case 'frame':
            setLastFrame(msg.jpeg);
            break;
          case 'score':
            setScore(msg.data);
            break;
          case 'event':
            setEvents(prev => [msg.data, ...prev]);
            break;
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => ws.close();
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, lastFrame, score, events, send };
}
```

- [ ] **Step 2: Create CameraFeed component**

Create `src/components/CameraFeed.tsx`:

```typescript
import React, { useRef, useEffect } from 'react';

interface Props {
  frameBase64: string | null;
  children?: React.ReactNode;
}

const CameraFeed: React.FC<Props> = ({ frameBase64, children }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!frameBase64 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frameBase64}`;
  }, [frameBase64]);

  return (
    <div style={{ position: 'relative', height: '100%', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {frameBase64 ? (
        <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
      ) : (
        <div style={{ color: '#555', fontSize: 14 }}>Waiting for camera feed...</div>
      )}
      {children}
    </div>
  );
};

export default CameraFeed;
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useWebSocket.ts frontend/src/components/CameraFeed.tsx
git commit -m "feat: useWebSocket hook with auto-reconnect + CameraFeed canvas component"
```

---

### Task 10: Live View Page

**Files:**
- Rewrite: `frontend/src/pages/LiveView.tsx`

- [ ] **Step 1: Implement Live View page**

Rewrite `src/pages/LiveView.tsx`:

```typescript
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
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
      {/* Status bar */}
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

      {/* Score correction dialog */}
      {showCorrect && (
        <div style={{ padding: '8px 20px', background: '#fffbeb', borderBottom: '1px solid #fdcb6e', display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 13 }}>Award point to:</span>
          <button className="btn" style={{ background: '#74b9ff', color: 'white', fontSize: 12 }} onClick={() => handleCorrect(1)}>Team A</button>
          <button className="btn" style={{ background: '#e17055', color: 'white', fontSize: 12 }} onClick={() => handleCorrect(2)}>Team B</button>
          <button style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 12 }} onClick={() => setShowCorrect(false)}>Cancel</button>
        </div>
      )}

      {/* Main area */}
      <PanelGroup direction="horizontal" style={{ flex: 1 }}>
        <Panel defaultSize={65} minSize={40}>
          <CameraFeed frameBase64={ws.lastFrame}>
            {ws.score && (
              <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>
                <Scoreboard score={ws.score} variant="overlay" />
              </div>
            )}
          </CameraFeed>
        </Panel>

        <PanelResizeHandle style={{ width: 4, background: '#e0e0e0', cursor: 'col-resize' }} />

        <Panel defaultSize={35} minSize={20}>
          <PanelGroup direction="vertical">
            <Panel defaultSize={50} minSize={20}>
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Scoreboard score={ws.score} />
                <EventLog events={ws.events} autoScroll />
              </div>
            </Panel>
            <PanelResizeHandle style={{ height: 4, background: '#e0e0e0', cursor: 'row-resize' }} />
            <Panel defaultSize={50} minSize={15}>
              <CourtMiniMap height={undefined} />
            </Panel>
          </PanelGroup>
        </Panel>
      </PanelGroup>
    </div>
  );
};

export default LiveView;
```

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: Success

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/LiveView.tsx
git commit -m "feat: Live View page with WebSocket feed, resizable panels, score correction"
```

---

### Task 11: Final Build Verification + Cleanup

**Files:**
- Verify all files build correctly

- [ ] **Step 1: Full build**

Run:
```bash
cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build
```
Expected: Build succeeds

- [ ] **Step 2: Run backend tests to confirm no regressions**

Run:
```bash
cd /Users/jonathan/Documents/Github/padel_analyzer && source venv/bin/activate && cd backend && python -m pytest tests/ -v --tb=short
```
Expected: All 133 tests pass

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Phase 3 frontend rebuild complete — all pages functional"
```
