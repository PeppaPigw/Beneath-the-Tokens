import React, {useEffect, useState} from 'react';

const STORAGE_KEY = 'beneath-the-tokens-font-scale';
const SCALES = [0.9, 1, 1.1, 1.2] as const;

export default function Root({children}: {children: React.ReactNode}) {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const stored = Number.parseFloat(window.localStorage.getItem(STORAGE_KEY) ?? '1');
    setScale(SCALES.find((value) => value === stored) ?? 1);
  }, []);

  useEffect(() => {
    document.documentElement.style.setProperty('--book-font-scale', String(scale));
    window.localStorage.setItem(STORAGE_KEY, String(scale));
  }, [scale]);

  const changeScale = (delta: number) => {
    setScale((current) => {
      const index = SCALES.indexOf(current as (typeof SCALES)[number]);
      return SCALES[Math.min(SCALES.length - 1, Math.max(0, index + delta))];
    });
  };

  return <>
    {children}
    <div className="fontScaleControl" role="group" aria-label="字号调整">
      <button type="button" onClick={() => changeScale(-1)} disabled={scale === SCALES[0]} aria-label="减小字号">A−</button>
      <button type="button" onClick={() => setScale(1)} aria-label="恢复默认字号">A</button>
      <button type="button" onClick={() => changeScale(1)} disabled={scale === SCALES[SCALES.length - 1]} aria-label="增大字号">A+</button>
    </div>
  </>;
}
