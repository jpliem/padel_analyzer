export interface MatchSummary {
  match_id: string;
  match_name: string;
  status: string;
  media?: MediaInfo | null;
  created_at?: string;
}

export interface MatchData {
  match_id: string;
  match_name: string;
  players: Record<string, string>;
  teams: Record<string, string[]>;
  golden_point: boolean;
  format: string;
  calibration: any | null;
  media?: MediaInfo | null;
  created_at?: string;
}

export interface MatchSetupData {
  match_name: string;
  players: Record<string, string>;
  teams: Record<string, string[]>;
  golden_point: boolean;
  format: string;
  out_of_court_play_enabled?: boolean;
  first_server?: string;
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

export interface ReviewRecord {
  id: string;
  frame_number: number;
  winner_team: number | null;
  reason: string;
  confidence: number;
  source: string;
  status: 'proposed' | 'confirmed' | 'rejected' | 'superseded';
  supersedes?: string | null;
  note?: string;
}

export interface AnalysisStatus {
  state: string;
  percent: number;
  match_id?: string;
  error?: string;
}

export interface MediaInfo {
  original_name: string;
  fps: number;
  frame_count: number;
  duration_seconds: number;
  width: number;
  height: number;
  size_bytes: number;
  uploaded_at: string;
  content_type?: string;
}

export interface Highlight {
  id: string;
  title: string;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  end_reason: string;
  confidence: number;
  needs_review: boolean;
}

export interface MatchStats {
  rallies: number;
  total_events: number;
  hits: number;
  bounces: number;
  wall_hits: number;
  serves: number;
  faults: number;
  average_rally_seconds: number;
  longest_rally_seconds: number;
  ball_track_points: number;
  frames_processed: number;
  pending_reviews: number;
}

export interface AnalysisResult {
  score: ScoreData;
  events: EventData[];
  trajectory: TrajectoryPoint[];
  player_positions: any[];
  reviews: ReviewRecord[];
  highlights: Highlight[];
  stats: MatchStats;
  media: MediaInfo;
  model_scope: 'single_camera';
  accuracy_notice: string;
  model_info?: {
    id: string;
    status: string;
    checkpoint?: string;
    selection_reason?: string;
    evidence?: {
      dataset: string;
      evaluation_split: string;
      visible_labels: number;
      matched_labels: number;
      precision: number;
      recall: number;
      tolerance_px: number;
    } | null;
    limitations?: string[];
  } | null;
  active_ball_diagnostics?: {
    total_candidates?: number;
    rejected_candidates?: number;
    uncertain_frames?: number;
  };
  evidence_status?: {
    audio?: { status?: string; warning?: string };
    audio_impulses?: number;
    pose?: string;
    contact_proposals?: number;
    semantic_rule_decisions?: number;
    scoring_policy?: string;
  };
  contact_proposals?: Array<{
    frame_number: number;
    timestamp: number;
    contact_type: string;
    confidence: number;
    requires_review: boolean;
    evidence: string[];
  }>;
  system_scope?: {
    runtime?: string[];
    research_only?: string[];
    not_validated?: string[];
  };
}

export interface MatchResult {
  match: MatchData;
  analysis: AnalysisResult | null;
  job: AnalysisStatus;
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
