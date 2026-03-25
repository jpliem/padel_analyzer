import type {
  MatchSummary, MatchData, MatchSetupData, ScoreData,
  EventData, AnalysisStatus, TrajectoryPoint, LiveStartData,
} from './types';

const API = 'http://localhost:8000';

export function getAnnotatedVideoUrl(matchId: string): string {
  return `${API}/match/${matchId}/annotated`;
}

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

export async function uploadVideo(matchId: string, file: File, detectorType: string = 'tracknet'): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  return fetchJSON(`${API}/analyze/upload?match_id=${matchId}&detector_type=${detectorType}`, {
    method: 'POST',
    body: form,
  });
}

export async function startAnalysis(jobId: string): Promise<void> {
  await fetchJSON(`${API}/analyze/start/${jobId}`, { method: 'POST' });
}

export async function startMatchAnalysis(matchId: string): Promise<void> {
  await fetchJSON(`${API}/match/${matchId}/analyze`, { method: 'POST' });
}

export async function getAnalysisStatus(jobId: string): Promise<AnalysisStatus> {
  return fetchJSON(`${API}/analyze/status/${jobId}`);
}

export async function getPositions(id: string): Promise<{ positions: any[] }> {
  return fetchJSON(`${API}/match/${id}/positions`);
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

export async function deleteAnalysis(matchId: string): Promise<void> {
  await fetchJSON(`${API}/match/${matchId}/analysis`, { method: 'DELETE' });
}

// Templates
export interface CalibrationTemplate {
  id: string;
  name: string;
  corners: number[][];
  net_points?: number[][] | null;
  thumbnail?: string | null;
  has_thumbnail?: boolean;
}

export async function listTemplates(): Promise<{ templates: CalibrationTemplate[] }> {
  return fetchJSON(`${API}/templates`);
}

export async function getTemplate(id: string): Promise<CalibrationTemplate> {
  return fetchJSON(`${API}/templates/${id}`);
}

export async function saveTemplate(data: {
  name: string; corners: number[][]; net_points?: number[][] | null; thumbnail?: string | null;
}): Promise<{ id: string }> {
  return fetchJSON(`${API}/templates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function deleteTemplate(id: string): Promise<void> {
  await fetchJSON(`${API}/templates/${id}`, { method: 'DELETE' });
}
