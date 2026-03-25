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
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (!url) return;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        switch (msg.type) {
          case 'frame': setLastFrame(msg.jpeg); break;
          case 'score': setScore(msg.data); break;
          case 'event': setEvents(prev => [msg.data, ...prev]); break;
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
