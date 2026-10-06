import React, {useEffect, useState, type ReactNode} from 'react';
import {useDocsSidebar} from '@docusaurus/plugin-content-docs/client';
import BackToTopButton from '@theme/BackToTopButton';
import DocRootLayoutSidebar from '@theme/DocRoot/Layout/Sidebar';
import DocRootLayoutMain from '@theme/DocRoot/Layout/Main';

import {SIDEBAR_PANEL_ID, TOC_PANEL_ID} from '../../readingLayout';

import styles from './styles.module.css';

export {SIDEBAR_PANEL_ID, TOC_PANEL_ID} from '../../readingLayout';

const STORAGE_KEYS = {
  sidebar: 'btt:reading-layout:sidebar-collapsed:v1',
  toc: 'btt:reading-layout:toc-collapsed:v1',
} as const;

export type PanelName = 'sidebar' | 'toc';

export interface ReadingLayoutState {
  sidebarCollapsed: boolean;
  tocCollapsed: boolean;
}

export interface Props {
  readonly children?: ReactNode;
}

/** Keep the two panel transitions independent and easy to smoke-test. */
export function togglePanel(
  state: ReadingLayoutState,
  panel: PanelName,
): ReadingLayoutState {
  if (panel === 'sidebar') {
    return {...state, sidebarCollapsed: !state.sidebarCollapsed};
  }
  return {...state, tocCollapsed: !state.tocCollapsed};
}

function readCollapsedPreference(key: string): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  try {
    return window.localStorage.getItem(key) === 'true';
  } catch {
    // Private browsing and blocked storage should never break reading.
    return false;
  }
}

function writeCollapsedPreference(key: string, value: boolean): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    // Storage is an enhancement; rendering and controls still work without it.
  }
}

export interface ReadingLayoutControlsProps {
  readonly sidebarCollapsed: boolean;
  readonly tocCollapsed: boolean;
  readonly onSidebarToggle: () => void;
  readonly onTocToggle: () => void;
  readonly sidebarAvailable?: boolean;
  readonly tocAvailable?: boolean;
}

export function ReadingLayoutControls({
  sidebarCollapsed,
  tocCollapsed,
  onSidebarToggle,
  onTocToggle,
  sidebarAvailable = true,
  tocAvailable = true,
}: ReadingLayoutControlsProps): ReactNode {
  return (
    <div className={styles.controls} role="group" aria-label="Reading layout controls">
      {sidebarAvailable && (
        <button
          className={styles.control}
          type="button"
          aria-controls={SIDEBAR_PANEL_ID}
          aria-label={`${sidebarCollapsed ? 'Expand' : 'Collapse'} sidebar`}
          aria-expanded={!sidebarCollapsed}
          onClick={onSidebarToggle}
        >
          {sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
        </button>
      )}
      {tocAvailable && (
        <button
          className={styles.control}
          type="button"
          aria-controls={TOC_PANEL_ID}
          aria-label={`${tocCollapsed ? 'Expand' : 'Collapse'} table of contents`}
          aria-expanded={!tocCollapsed}
          onClick={onTocToggle}
        >
          {tocCollapsed ? 'Show table of contents' : 'Hide table of contents'}
        </button>
      )}
    </div>
  );
}

export interface ReadingLayoutFrameProps extends ReadingLayoutControlsProps {
  readonly children?: ReactNode;
}

export function ReadingLayoutFrame({
  children,
  sidebarCollapsed,
  tocCollapsed,
  onSidebarToggle,
  onTocToggle,
  sidebarAvailable = true,
  tocAvailable = true,
}: ReadingLayoutFrameProps): ReactNode {
  return (
    <div
      className={styles.docsWrapper}
      data-btt-sidebar-collapsed={sidebarCollapsed}
      data-btt-toc-collapsed={tocCollapsed}
    >
      <ReadingLayoutControls
        sidebarCollapsed={sidebarCollapsed}
        tocCollapsed={tocCollapsed}
        onSidebarToggle={onSidebarToggle}
        onTocToggle={onTocToggle}
        sidebarAvailable={sidebarAvailable}
      />
      <div className={styles.content}>{children}</div>
    </div>
  );
}

/**
 * Docusaurus 3's DocRoot/Layout boundary with a small reading control layer.
 * State starts expanded for SSR/hydration parity. Preferences are read in an
 * effect after the first client render and persisted only after that read.
 */
export default function DocRootLayout({children}: Props): ReactNode {
  const sidebar = useDocsSidebar();
  const [hiddenSidebarContainer, setHiddenSidebarContainer] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [tocCollapsed, setTocCollapsed] = useState(false);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);

  useEffect(() => {
    setSidebarCollapsed(readCollapsedPreference(STORAGE_KEYS.sidebar));
    setTocCollapsed(readCollapsedPreference(STORAGE_KEYS.toc));
    setPreferencesLoaded(true);
  }, []);

  useEffect(() => {
    if (preferencesLoaded) {
      writeCollapsedPreference(STORAGE_KEYS.sidebar, sidebarCollapsed);
    }
  }, [preferencesLoaded, sidebarCollapsed]);

  useEffect(() => {
    if (preferencesLoaded) {
      writeCollapsedPreference(STORAGE_KEYS.toc, tocCollapsed);
    }
  }, [preferencesLoaded, tocCollapsed]);

  return (
    <ReadingLayoutFrame
      sidebarCollapsed={sidebarCollapsed}
      tocCollapsed={tocCollapsed}
      onSidebarToggle={() => setSidebarCollapsed((current) => !current)}
      onTocToggle={() => setTocCollapsed((current) => !current)}
      sidebarAvailable={Boolean(sidebar)}
    >
      <BackToTopButton />
      <div className={styles.docRoot}>
        {sidebar && (
          <div id={SIDEBAR_PANEL_ID} className={styles.sidebarPanel}>
            <DocRootLayoutSidebar
              sidebar={sidebar.items}
              hiddenSidebarContainer={hiddenSidebarContainer}
              setHiddenSidebarContainer={setHiddenSidebarContainer}
            />
          </div>
        )}
        <DocRootLayoutMain hiddenSidebarContainer={hiddenSidebarContainer}>
          {children}
        </DocRootLayoutMain>
      </div>
    </ReadingLayoutFrame>
  );
}
