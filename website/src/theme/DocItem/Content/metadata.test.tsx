import React from 'react';
import assert from 'node:assert/strict';
import {renderToStaticMarkup} from 'react-dom/server';
import {ChapterMetadata} from './index';

function render(frontMatter?: Record<string, unknown>): string {
  return renderToStaticMarkup(<ChapterMetadata frontMatter={frontMatter} />);
}

export function runMetadataSmokeTest(): void {
  const full = render({
    level: 'systems',
    estimated_hours: 12,
    prerequisites: ['ch01', 'ch02'],
    paper_count: 4,
    source_commit: 'abc1234',
    lab_path: 'labs/chapter-01',
    last_verified: '2026-10-05',
  });
  for (const key of ['level', 'estimated_hours', 'prerequisites', 'paper_count', 'source_commit', 'lab_path', 'last_verified']) {
    assert.ok(full.includes(`data-metadata-key="${key}"`), `missing metadata key: ${key}`);
  }
  assert.match(full, /Chapter metadata/);
  assert.match(full, /12 小时/);
  assert.match(full, /ch01, ch02/);
  assert.match(full, /abc1234/);

  const partial = render({estimated_hours: 0, paper_count: 0, source_commit: '  '});
  assert.match(partial, /data-metadata-key="estimated_hours"/);
  assert.match(partial, /data-metadata-key="paper_count"/);
  assert.doesNotMatch(partial, /data-metadata-key="level"/);
  assert.doesNotMatch(partial, /data-metadata-key="source_commit"/);
  assert.doesNotMatch(partial, /前置章节/);

  assert.equal(render(), '');
  assert.equal(render({prerequisites: [], level: '', lab_path: null}), '');
}
