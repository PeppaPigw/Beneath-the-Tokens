import {createContext, useContext} from 'react';

/** Stable DOM ids shared by the reading controls and their panel regions. */
export const SIDEBAR_PANEL_ID = 'btt-doc-sidebar';
export const TOC_PANEL_ID = 'btt-doc-toc';
/** The Docusaurus mobile TOC is rendered separately from its desktop TOC. */
export const MOBILE_TOC_PANEL_ID = 'btt-doc-toc-mobile';

export interface ReadingLayoutAvailability {
  tocAvailable: boolean;
  setTocAvailable: (available: boolean) => void;
}

export const ReadingLayoutAvailabilityContext = createContext<ReadingLayoutAvailability | null>(null);

export function useReadingLayoutAvailability(): ReadingLayoutAvailability | null {
  return useContext(ReadingLayoutAvailabilityContext);
}
