import React from 'react';
import clsx from 'clsx';
import IconArrow from '@theme/Icon/Arrow';

export default function CollapseButton({onClick}: {onClick: () => void}) {
  return (
    <button type="button" className={clsx('button', 'button--secondary', 'button--outline', 'sidebarArrowButton')}
      onClick={onClick} title="折叠教材目录" aria-label="折叠教材目录">
      <IconArrow className="sidebarArrowIcon" />
    </button>
  );
}
