import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

interface PlayerDot {
  id: string;
  x: number;
  y: number;
  team: 'A' | 'B';
  label?: string;
}

interface Props {
  players?: PlayerDot[];
  ballPosition?: { x: number; y: number; z?: number } | null;
  ballTrail?: { x: number; y: number; z?: number }[];
}

// Court: 10m wide (X), 20m long (Z in Three.js), Y is up
const CX = (x: number) => x - 5;     // center X
const CZ = (y: number) => -(y - 10); // flip Y to Z, center
const CY = (z?: number) => z || 0;    // height

const CourtSurface: React.FC = () => (
  <group>
    {/* Court surface */}
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <planeGeometry args={[10, 20]} />
      <meshStandardMaterial color="#1565C0" />
    </mesh>

    {/* Court lines */}
    {/* Court lines as thin boxes */}
    {[
      { from: [-5, 10], to: [5, 10] },     // near baseline
      { from: [-5, -10], to: [5, -10] },    // far baseline
      { from: [-5, -10], to: [-5, 10] },    // left sideline
      { from: [5, -10], to: [5, 10] },      // right sideline
      { from: [-5, CZ(6.95)], to: [5, CZ(6.95)] },   // near service
      { from: [-5, CZ(13.05)], to: [5, CZ(13.05)] },  // far service
      { from: [0, CZ(6.95)], to: [0, CZ(13.05)] },    // center service
    ].map((l, i) => {
      const mx = (l.from[0] + l.to[0]) / 2;
      const mz = (l.from[1] + l.to[1]) / 2;
      const dx = Math.abs(l.to[0] - l.from[0]);
      const dz = Math.abs(l.to[1] - l.from[1]);
      return (
        <mesh key={`line-${i}`} position={[mx, 0.02, mz]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[Math.max(dx, 0.05), Math.max(dz, 0.05)]} />
          <meshStandardMaterial color="white" />
        </mesh>
      );
    })}

    {/* Net */}
    <mesh position={[0, 0.44, 0]}>
      <boxGeometry args={[10, 0.88, 0.05]} />
      <meshStandardMaterial color="white" transparent opacity={0.4} />
    </mesh>

    {/* Net posts */}
    <mesh position={[-5, 0.46, 0]}>
      <cylinderGeometry args={[0.04, 0.04, 0.92]} />
      <meshStandardMaterial color="#888" />
    </mesh>
    <mesh position={[5, 0.46, 0]}>
      <cylinderGeometry args={[0.04, 0.04, 0.92]} />
      <meshStandardMaterial color="#888" />
    </mesh>

    {/* Glass walls (transparent) */}
    {[
      { pos: [0, 1.5, 10.1] as [number, number, number], size: [10, 3, 0.1] as [number, number, number] },
      { pos: [0, 1.5, -10.1] as [number, number, number], size: [10, 3, 0.1] as [number, number, number] },
      { pos: [-5.1, 1.5, 0] as [number, number, number], size: [0.1, 3, 20] as [number, number, number] },
      { pos: [5.1, 1.5, 0] as [number, number, number], size: [0.1, 3, 20] as [number, number, number] },
    ].map((wall, i) => (
      <mesh key={`wall-${i}`} position={wall.pos}>
        <boxGeometry args={wall.size} />
        <meshStandardMaterial color="#88ccff" transparent opacity={0.08} />
      </mesh>
    ))}
  </group>
);

const PlayerFigure: React.FC<{ position: [number, number, number]; color: string; label: string }> = ({ position, color, label }) => (
  <group position={position}>
    {/* Body */}
    <mesh position={[0, 0.5, 0]}>
      <capsuleGeometry args={[0.15, 0.6, 4, 8]} />
      <meshStandardMaterial color={color} />
    </mesh>
    {/* Head */}
    <mesh position={[0, 0.95, 0]}>
      <sphereGeometry args={[0.12]} />
      <meshStandardMaterial color="#ffeaa7" />
    </mesh>
    {/* Label */}
    <Text position={[0, 1.3, 0]} fontSize={0.3} color={color}
      anchorX="center" anchorY="middle">{label}</Text>
  </group>
);

const BallMesh: React.FC<{ position: [number, number, number] }> = ({ position }) => {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.x += delta * 5;
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.065]} />
      <meshStandardMaterial color="#fdcb6e" emissive="#fdcb6e" emissiveIntensity={0.5} />
    </mesh>
  );
};

const Court3DView: React.FC<Props> = ({ players = [], ballPosition, ballTrail = [] }) => {
  // Deduplicate players by ID — only keep the first occurrence
  const uniquePlayers = players.filter((p, i, arr) => arr.findIndex(x => x.id === p.id) === i);

  return (
  <div style={{ width: '100%', height: '100%', background: '#0a1628' }}>
    <Canvas camera={{ position: [0, 18, 14], fov: 45 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={0.6} />
      <directionalLight position={[-10, 15, -10]} intensity={0.3} />

      <CourtSurface />

      {/* Players — max 4 */}
      {uniquePlayers.slice(0, 4).map(p => (
        <PlayerFigure
          key={p.id}
          position={[CX(p.x), 0, CZ(p.y)]}
          color={p.team === 'A' ? '#74b9ff' : '#e17055'}
          label={p.label || p.id}
        />
      ))}

      {/* Ball trail */}
      {ballTrail.map((p, i) => (
        <mesh key={`trail-${i}`} position={[CX(p.x), CY(p.z), CZ(p.y)]}>
          <sphereGeometry args={[0.03]} />
          <meshStandardMaterial
            color="#fdcb6e"
            transparent
            opacity={0.2 + (i / Math.max(ballTrail.length - 1, 1)) * 0.6}
          />
        </mesh>
      ))}

      {/* Ball */}
      {ballPosition && (
        <>
          <BallMesh position={[CX(ballPosition.x), CY(ballPosition.z), CZ(ballPosition.y)]} />
          {/* Shadow on ground */}
          <mesh position={[CX(ballPosition.x), 0.01, CZ(ballPosition.y)]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.1]} />
            <meshStandardMaterial color="black" transparent opacity={0.3} />
          </mesh>
          {/* Height line (thin cylinder from ground to ball) */}
          {(ballPosition.z || 0) > 0.1 && (
            <mesh position={[CX(ballPosition.x), CY(ballPosition.z) / 2, CZ(ballPosition.y)]}>
              <cylinderGeometry args={[0.01, 0.01, CY(ballPosition.z), 4]} />
              <meshStandardMaterial color="#fdcb6e" transparent opacity={0.4} />
            </mesh>
          )}
        </>
      )}

      <OrbitControls
        enablePan={true}
        enableZoom={true}
        maxPolarAngle={Math.PI / 2.1}
        target={[0, 0, 0]}
      />
    </Canvas>
  </div>
  );
};

export default Court3DView;
