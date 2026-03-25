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
