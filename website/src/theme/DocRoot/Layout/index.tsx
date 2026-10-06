import React, {useEffect, useRef, useState, type ReactNode, type RefObject} from 'react';
import {useDocsSidebar} from '@docusaurus/plugin-content-docs/client';
import BackToTopButton from '@theme/BackToTopButton';
import DocRootLayoutSidebar from '@theme/DocRoot/Layout/Sidebar';
import DocRootLayoutMain from '@theme/DocRoot/Layout/Main';
import DocSidebarItems from '@theme/DocSidebarItems';
import {useLocation} from '@docusaurus/router';

import {
  SIDEBAR_PANEL_ID,
  TOC_PANEL_ID,
  MOBILE_TOC_PANEL_ID,
  ReadingLayoutAvailabilityContext,
} from '../../readingLayout';

import styles from './styles.module.css';

export {SIDEBAR_PANEL_ID, TOC_PANEL_ID, MOBILE_TOC_PANEL_ID} from '../../readingLayout';

export const STORAGE_KEYS = {
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

export function closeMobileDrawers(
  state: ReadingLayoutState,
): ReadingLayoutState {
  return {...state, sidebarCollapsed: true, tocCollapsed: true};
}

export function escapeFocusPanel(
  tocAvailable: boolean,
  tocCollapsed: boolean,
): PanelName {
  return tocAvailable && !tocCollapsed ? 'toc' : 'sidebar';
}

/** Parse only values written by this layout. Everything else is a safe default. */
export function parseCollapsedPreference(value: string | null): boolean {
  return value === 'true';
}

export function readCollapsedPreference(key: string): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  try {
    return parseCollapsedPreference(window.localStorage.getItem(key));
  } catch {
    // Private browsing and blocked storage should never break reading.
    return false;
  }
}

export function writeCollapsedPreference(key: string, value: boolean): void {
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
  readonly sidebarToggleRef?: RefObject<HTMLButtonElement | null>;
  readonly tocToggleRef?: RefObject<HTMLButtonElement | null>;
}

export function ReadingLayoutControls({
  sidebarCollapsed,
  tocCollapsed,
  onSidebarToggle,
  onTocToggle,
  sidebarAvailable = true,
  tocAvailable = true,
  sidebarToggleRef,
  tocToggleRef,
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
          ref={sidebarToggleRef}
        >
          {sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
        </button>
      )}
      {tocAvailable && (
        <button
          className={styles.control}
          type="button"
          aria-controls={`${TOC_PANEL_ID} ${MOBILE_TOC_PANEL_ID}`}
          aria-label={`${tocCollapsed ? 'Expand' : 'Collapse'} table of contents`}
          aria-expanded={!tocCollapsed}
          onClick={onTocToggle}
          ref={tocToggleRef}
        >
          {tocCollapsed ? 'Show table of contents' : 'Hide table of contents'}
        </button>
      )}
    </div>
  );
}

export interface ReadingLayoutFrameProps extends ReadingLayoutControlsProps {
  readonly children?: ReactNode;
  readonly layoutReady?: boolean;
}

export function ReadingLayoutFrame({
  children,
  sidebarCollapsed,
  tocCollapsed,
  onSidebarToggle,
  onTocToggle,
  sidebarAvailable = true,
  tocAvailable = true,
  sidebarToggleRef,
  tocToggleRef,
  layoutReady = false,
}: ReadingLayoutFrameProps): ReactNode {
  return (
    <div
      className={styles.docsWrapper}
      data-btt-sidebar-collapsed={sidebarCollapsed}
      data-btt-toc-collapsed={tocCollapsed}
      data-btt-layout-ready={layoutReady}
    >
      <ReadingLayoutControls
        sidebarCollapsed={sidebarCollapsed}
        tocCollapsed={tocCollapsed}
        onSidebarToggle={onSidebarToggle}
        onTocToggle={onTocToggle}
        sidebarAvailable={sidebarAvailable}
        tocAvailable={tocAvailable}
        sidebarToggleRef={sidebarToggleRef}
        tocToggleRef={tocToggleRef}
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
  const {pathname} = useLocation();
  const [hiddenSidebarContainer, setHiddenSidebarContainer] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [tocCollapsed, setTocCollapsed] = useState(false);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [tocAvailable, setTocAvailable] = useState(false);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);
  const tocToggleRef = useRef<HTMLButtonElement>(null);

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

  // A mobile drawer must always have an escape hatch. Keep this listener at the
  // document level so Escape works while focus is inside a panel link.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || (sidebarCollapsed && tocCollapsed)) {
        return;
      }
      const mobile =
        typeof window !== 'undefined' &&
        (window.matchMedia?.('(max-width: 996px)').matches ??
          window.innerWidth <= 996);
      if (!mobile) {
        return;
      }
      event.preventDefault();
      const focusTarget = escapeFocusPanel(tocAvailable, tocCollapsed) === 'toc'
        ? tocToggleRef.current
        : sidebarToggleRef.current;
      const closed = closeMobileDrawers({sidebarCollapsed, tocCollapsed});
      setSidebarCollapsed(closed.sidebarCollapsed);
      setTocCollapsed(closed.tocCollapsed);
      focusTarget?.focus();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [sidebarCollapsed, tocAvailable, tocCollapsed]);

  return (
    <ReadingLayoutAvailabilityContext.Provider value={{tocAvailable, setTocAvailable}}>
    <ReadingLayoutFrame
      sidebarCollapsed={sidebarCollapsed}
      tocCollapsed={tocCollapsed}
      onSidebarToggle={() =>
          setSidebarCollapsed((current) => !current)
      }
      onTocToggle={() => setTocCollapsed((current) => !current)}
      sidebarAvailable={Boolean(sidebar)}
      tocAvailable={tocAvailable}
      sidebarToggleRef={sidebarToggleRef}
      tocToggleRef={tocToggleRef}
      layoutReady={preferencesLoaded}
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
            <nav className={styles.mobileSidebar} aria-label="Docs sidebar">
              <ul className="menu__list">
                <DocSidebarItems
                  items={sidebar.items}
                  activePath={pathname}
                  level={1}
                  onItemClick={() => setSidebarCollapsed(true)}
                />
              </ul>
            </nav>
          </div>
        )}
        <div className={styles.mainPanel}>
          <DocRootLayoutMain hiddenSidebarContainer={hiddenSidebarContainer}>
            {children}
          </DocRootLayoutMain>
        </div>
      </div>
    </ReadingLayoutFrame>
    </ReadingLayoutAvailabilityContext.Provider>
  );
}
