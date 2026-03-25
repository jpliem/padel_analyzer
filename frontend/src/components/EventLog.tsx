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
            <div key={i} onClick={() => onEventClick?.(event)}
              style={{
                display: 'flex', gap: 8, padding: '6px 8px', borderRadius: 6, fontSize: 12,
                background: isPoint ? '#fffbeb' : 'white',
                border: isPoint ? '1px solid #fdcb6e' : '1px solid #e8e8e8',
                cursor: onEventClick ? 'pointer' : 'default',
              }}>
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
