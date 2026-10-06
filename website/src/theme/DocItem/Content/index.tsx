import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import Heading from '@theme/Heading';
import MDXContent from '@theme/MDXContent';

/**
 * Docusaurus exposes front matter as an untyped record. Keep this boundary
 * deliberately permissive so old documents continue to render when they do
 * not have any of the learning metadata fields.
 */
export interface ChapterFrontMatter {
  readonly level?: unknown;
  readonly estimated_hours?: unknown;
  readonly prerequisites?: unknown;
  readonly paper_count?: unknown;
  readonly source_commit?: unknown;
  readonly lab_path?: unknown;
  readonly last_verified?: unknown;
}

interface MetadataEntry {
  readonly key: keyof ChapterFrontMatter;
  readonly label: string;
  readonly value: string;
}

function nonEmptyText(value: unknown): string | null {
  if (typeof value === 'string') {
    const text = value.trim();
    return text.length > 0 ? text : null;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }
  return null;
}

function nonNegativeCount(value: unknown): string | null {
  const text = nonEmptyText(value);
  if (text === null) {
    return null;
  }
  const count = Number(text);
  return Number.isFinite(count) && count >= 0 ? text : null;
}

function prerequisitesText(value: unknown): string | null {
  if (!Array.isArray(value)) {
    return nonEmptyText(value);
  }
  const items = value
    .map((item) => nonEmptyText(item))
    .filter((item): item is string => item !== null);
  return items.length > 0 ? items.join(', ') : null;
}

function metadataEntries(frontMatter: ChapterFrontMatter): MetadataEntry[] {
  const entries: MetadataEntry[] = [];
  const level = nonEmptyText(frontMatter.level);
  if (level !== null) {
    entries.push({key: 'level', label: '学习级别', value: level});
  }

  const hours = nonNegativeCount(frontMatter.estimated_hours);
  if (hours !== null) {
    entries.push({key: 'estimated_hours', label: '预计学习时间', value: `${hours} 小时`});
  }

  const prerequisites = prerequisitesText(frontMatter.prerequisites);
  if (prerequisites !== null) {
    entries.push({key: 'prerequisites', label: '前置章节', value: prerequisites});
  }

  const paperCount = nonNegativeCount(frontMatter.paper_count);
  if (paperCount !== null) {
    entries.push({key: 'paper_count', label: '论文数量', value: paperCount});
  }

  const sourceCommit = nonEmptyText(frontMatter.source_commit);
  if (sourceCommit !== null) {
    entries.push({key: 'source_commit', label: '源码提交', value: sourceCommit});
  }

  const labPath = nonEmptyText(frontMatter.lab_path);
  if (labPath !== null) {
    entries.push({key: 'lab_path', label: '实验路径', value: labPath});
  }

  const lastVerified = nonEmptyText(frontMatter.last_verified);
  if (lastVerified !== null) {
    entries.push({key: 'last_verified', label: '最近核验', value: lastVerified});
  }

  return entries;
}

/** Render only fields that are present; an entirely unannotated document has no empty chrome. */
export function ChapterMetadata({frontMatter}: {readonly frontMatter?: ChapterFrontMatter}): ReactNode {
  const entries = metadataEntries(frontMatter ?? {});
  if (entries.length === 0) {
    return null;
  }

  return (
    <aside className="bttChapterMetadata" aria-label="Chapter metadata">
      <dl>
        {entries.map(({key, label, value}) => (
          <div className="bttChapterMetadata__item" data-metadata-key={key} key={key}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

/**
 * Preserve Docusaurus's stock Content boundary and Markdown renderer. Metadata
 * is a sibling of MDXContent, so article headings, links, and source order keep
 * their usual semantics.
 */
export default function DocItemContent({children}: {readonly children?: ReactNode}): ReactNode {
  const {metadata, frontMatter, contentTitle} = useDoc();
  const shouldRenderSyntheticTitle = !frontMatter.hide_title && typeof contentTitle === 'undefined';
  const syntheticTitle = shouldRenderSyntheticTitle ? metadata.title : null;

  return (
    <div className={clsx(ThemeClassNames.docs.docMarkdown, 'markdown')}>
      {syntheticTitle && (
        <header>
          <Heading as="h1">{syntheticTitle}</Heading>
        </header>
      )}
      {/*
       * A document with a synthetic title has its heading here, so metadata
       * can follow it immediately. For documents with an explicit Markdown
       * H1, MDXContent owns that heading; place metadata after that rendered
       * content rather than before the title.
       */}
      {!contentTitle && <ChapterMetadata frontMatter={frontMatter as ChapterFrontMatter} />}
      <MDXContent>{children}</MDXContent>
      {contentTitle && <ChapterMetadata frontMatter={frontMatter as ChapterFrontMatter} />}
    </div>
  );
}
