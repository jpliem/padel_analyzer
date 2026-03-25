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
  rtsp_url?: string;
  record: boolean;
}
