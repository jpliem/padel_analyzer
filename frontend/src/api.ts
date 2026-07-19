import type {
  MatchSummary, MatchData, MatchSetupData, ScoreData,
  EventData, AnalysisStatus, TrajectoryPoint, LiveStartData, ReviewRecord,
  MatchResult, Highlight, MatchStats,
} from './types';

const API_PORT = process.env.REACT_APP_API_PORT || '8000';
const API = process.env.REACT_APP_API_URL || `http://${window.location.hostname || 'localhost'}:${API_PORT}`;

export async function autoDetectCourt(matchId: string): Promise<{ keypoints: number[][]; mode: string }> {
  return fetchJSON(`${API}/match/${matchId}/auto-detect-court`, { method: 'POST' });
}

export function getAnnotatedVideoUrl(matchId: string): string {
  return `${API}/match/${matchId}/annotated`;
}

export function getRecordingVideoUrl(matchId: string): string {
  return `${API}/match/${matchId}/recording`;
}

export function getExportUrl(matchId: string, format: 'json' | 'csv'): string {
  return `${API}/match/${matchId}/export.${format}`;
}

export function getHighlightClipUrl(matchId: string, highlightId: string): string {
  return `${API}/match/${matchId}/highlights/${highlightId}.mp4`;
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

export async function calibrate(id: string, corners: number[][], netPoints?: number[][] | null, netTopPoints?: number[][] | null): Promise<void> {
  await fetchJSON(`${API}/match/${id}/calibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      corners,
      net_points: netPoints || null,
      net_top_points: netTopPoints || null,
    }),
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

export async function cancelAnalysis(jobId: string): Promise<void> {
  await fetchJSON(`${API}/analyze/cancel/${jobId}`, { method: 'POST' });
}

export async function getMatchResult(id: string): Promise<MatchResult> {
  return fetchJSON(`${API}/match/${id}/result`);
}

export async function getHighlights(id: string): Promise<{ highlights: Highlight[] }> {
  return fetchJSON(`${API}/match/${id}/highlights`);
}

export async function getStats(id: string): Promise<{ stats: MatchStats }> {
  return fetchJSON(`${API}/match/${id}/stats`);
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

export async function getReviews(id: string): Promise<{ reviews: ReviewRecord[] }> {
  return fetchJSON(`${API}/match/${id}/reviews`);
}

export async function resolveReview(matchId: string, recordId: string,
  confirmed: boolean, winnerTeam?: number): Promise<{ review: ReviewRecord; score: ScoreData }> {
  return fetchJSON(`${API}/match/${matchId}/reviews/${recordId}/resolve`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirmed, winner_team: winnerTeam ?? null }),
  });
}

export async function correctReview(matchId: string, recordId: string,
  winnerTeam: number): Promise<{ review: ReviewRecord; score: ScoreData }> {
  return fetchJSON(`${API}/match/${matchId}/reviews/${recordId}/correct`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ winner_team: winnerTeam, reason: 'manual' }),
  });
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

export async function correctScore(matchId: string, team: number): Promise<{ score: ScoreData }> {
  return fetchJSON(`${API}/match/${matchId}/correct-score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ team }),
  });
}

export async function deleteAnalysis(matchId: string): Promise<void> {
  await fetchJSON(`${API}/match/${matchId}/analysis`, { method: 'DELETE' });
}

export async function deleteMatch(matchId: string): Promise<void> {
  await fetchJSON(`${API}/match/${matchId}`, { method: 'DELETE' });
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
