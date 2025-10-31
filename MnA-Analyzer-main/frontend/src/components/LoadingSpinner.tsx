import React from 'react';

export const LoadingSpinner: React.FC<{ size?: number }> = ({ size = 24 }) => {
  return (
    <div 
      style={{ 
        width: size, 
        height: size, 
        border: '2px solid #f3f3f3',
        borderTop: '2px solid #3498db',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
        display: 'inline-block',
        marginRight: '8px',
      }}
    />
  );
};

export default LoadingSpinner;