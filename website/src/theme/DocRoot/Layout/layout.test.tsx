/**
 * A dependency-free smoke test for the reading layout boundary.
 *
 * The website package intentionally has no test runner yet. This file is
 * compiled and executed by the phase-two CI smoke command (see the report
 * beside this task), while still being useful to Jest/Vitest consumers.
 */
import React from 'react';


import {renderToStaticMarkup} from 'react-dom/server';
import {
  SIDEBAR_PANEL_ID,
  TOC_PANEL_ID,
  MOBILE_TOC_PANEL_ID,
  ReadingLayoutControls,
  ReadingLayoutFrame,
  closeMobileDrawers,
  parseCollapsedPreference,
  readCollapsedPreference,
  writeCollapsedPreference,
  STORAGE_KEYS,
  togglePanel,
  type ReadingLayoutState,
} from './index';

type ControlButton = React.ReactElement<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    'aria-controls'?: string;
  }
>;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

export function runLayoutSmokeTest(): void {
  const markup = renderToStaticMarkup(
    <ReadingLayoutFrame
      sidebarCollapsed={false}
      tocCollapsed={false}
      onSidebarToggle={() => undefined}
      onTocToggle={() => undefined}
    >
      <aside id={SIDEBAR_PANEL_ID}>Sidebar fixture</aside>
      <aside id={TOC_PANEL_ID}>TOC fixture</aside>
      <aside id={MOBILE_TOC_PANEL_ID}>Mobile TOC fixture</aside>
    </ReadingLayoutFrame>,
  );

  assert(
    markup.includes(`aria-controls="${SIDEBAR_PANEL_ID}"`),
    'sidebar control should point at the stable sidebar id',
  );
  assert(
    markup.includes(`aria-controls="${TOC_PANEL_ID} ${MOBILE_TOC_PANEL_ID}"`),
    'TOC control should point at both desktop and mobile TOC ids',
  );
  assert(
    /aria-label="(?:Collapse|Expand) sidebar"/.test(markup),
    'sidebar control should expose an accessible name',
  );
  assert(
    /aria-label="(?:Collapse|Expand) table of contents"/.test(markup),
    'TOC control should expose an accessible name',
  );
  assert(
    markup.includes('data-btt-sidebar-collapsed="false"'),
    'SSR output should start with an expanded sidebar',
  );
  assert(
    markup.includes('data-btt-toc-collapsed="false"'),
    'SSR output should start with an expanded TOC',
  );

  const targets = [
    `id="${SIDEBAR_PANEL_ID}"`,
    `id="${TOC_PANEL_ID}"`,
    `id="${MOBILE_TOC_PANEL_ID}"`,
  ];
  for (const target of targets) {
    assert(markup.includes(target), `rendered panel target missing: ${target}`);
  }

  let renderedState: ReadingLayoutState = {
    sidebarCollapsed: false,
    tocCollapsed: false,
  };
  const controls = ReadingLayoutControls({
    sidebarCollapsed: renderedState.sidebarCollapsed,
    tocCollapsed: renderedState.tocCollapsed,
    onSidebarToggle: () => {
      renderedState = togglePanel(renderedState, 'sidebar');
    },
    onTocToggle: () => {
      renderedState = togglePanel(renderedState, 'toc');
    },
  }) as React.ReactElement<{children?: React.ReactNode}>;
  const buttons = React.Children.toArray(controls.props.children) as ControlButton[];
  const sidebarButton = buttons.find(
    (button) => button.props['aria-controls'] === SIDEBAR_PANEL_ID,
  );
  const tocButton = buttons.find(
    (button) => button.props['aria-controls']?.split(' ').includes(TOC_PANEL_ID),
  );
  assert(sidebarButton, 'rendered sidebar control should be present');
  assert(tocButton, 'rendered TOC control should be present');
  const clickSidebar = sidebarButton.props.onClick as (() => void) | undefined;
  assert(clickSidebar, 'sidebar control should have a click handler');
  clickSidebar();
  assert(renderedState.sidebarCollapsed, 'sidebar control should collapse its panel');
  assert(
    !renderedState.tocCollapsed,
    'sidebar control must leave the TOC expanded',
  );
  const clickToc = tocButton.props.onClick as (() => void) | undefined;
  assert(clickToc, 'TOC control should have a click handler');
  clickToc();
  assert(renderedState.tocCollapsed, 'TOC control should collapse its panel');
  assert(
    renderedState.sidebarCollapsed,
    'TOC control must leave the sidebar collapsed',
  );

  const noSidebarControls = ReadingLayoutControls({
    sidebarCollapsed: false,
    tocCollapsed: false,
    sidebarAvailable: false,
    onSidebarToggle: () => undefined,
    onTocToggle: () => undefined,
  }) as React.ReactElement<{children?: React.ReactNode}>;
  const noSidebarButtons = React.Children.toArray(
    noSidebarControls.props.children,
  ) as ControlButton[];
  assert(
    !noSidebarButtons.some(
      (button) => button.props['aria-controls'] === SIDEBAR_PANEL_ID,
    ),
    'sidebar control should be omitted when the sidebar target is unavailable',
  );

  const noTocControls = ReadingLayoutControls({
    sidebarCollapsed: false,
    tocCollapsed: false,
    tocAvailable: false,
    onSidebarToggle: () => undefined,
    onTocToggle: () => undefined,
  }) as React.ReactElement<{children?: React.ReactNode}>;
  const noTocButtons = React.Children.toArray(
    noTocControls.props.children,
  ) as ControlButton[];
  assert(
    !noTocButtons.some((button) =>
      button.props['aria-controls']?.split(' ').includes(TOC_PANEL_ID),
    ),
    'TOC control should be omitted when the TOC target is unavailable',
  );

  const initial: ReadingLayoutState = {
    sidebarCollapsed: false,
    tocCollapsed: false,
  };
  const afterSidebar = togglePanel(initial, 'sidebar');
  assert(afterSidebar.sidebarCollapsed, 'toggling the sidebar should collapse it');
  assert(
    !afterSidebar.tocCollapsed,
    'toggling the sidebar must leave the TOC expanded',
  );
  const afterToc = togglePanel(afterSidebar, 'toc');
  assert(afterToc.tocCollapsed, 'toggling the TOC should collapse it');
  assert(
    afterToc.sidebarCollapsed,
    'toggling the TOC must leave the sidebar collapsed',
  );

  // Preferences are intentionally independent and reject malformed values.
  assert(parseCollapsedPreference('true'), 'true should restore a collapsed panel');
  assert(!parseCollapsedPreference('false'), 'false should restore an expanded panel');
  assert(!parseCollapsedPreference('1'), 'non-boolean storage should use the expanded fallback');
  assert(!parseCollapsedPreference('{"collapsed":true}'), 'malformed storage should use the expanded fallback');
  const originalWindow = (globalThis as {window?: unknown}).window;
  const values = new Map<string, string>();
  const localStorage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
  };
  (globalThis as {window?: unknown}).window = {localStorage};
  writeCollapsedPreference(STORAGE_KEYS.sidebar, true);
  writeCollapsedPreference(STORAGE_KEYS.toc, false);
  assert(readCollapsedPreference(STORAGE_KEYS.sidebar), 'sidebar preference should persist independently');
  assert(!readCollapsedPreference(STORAGE_KEYS.toc), 'TOC preference should persist independently');
  values.set(STORAGE_KEYS.sidebar, 'not-json');
  assert(!readCollapsedPreference(STORAGE_KEYS.sidebar), 'malformed localStorage should fall back safely');
  (globalThis as {window?: unknown}).window = {
    localStorage: {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    },
  };
  assert(!readCollapsedPreference(STORAGE_KEYS.sidebar), 'blocked localStorage should fall back safely');
  writeCollapsedPreference(STORAGE_KEYS.sidebar, true);
  (globalThis as {window?: unknown}).window = originalWindow;
  const closed = closeMobileDrawers({sidebarCollapsed: false, tocCollapsed: true});
  assert(closed.sidebarCollapsed, 'Escape should close an open mobile sidebar drawer');
  assert(closed.tocCollapsed, 'Escape should leave the TOC drawer closed');
}
