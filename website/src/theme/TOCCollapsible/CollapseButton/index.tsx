import React from 'react';

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {collapsed: boolean};

export default function CollapseButton({collapsed, ...props}: Props) {
  return (
    <button type="button" {...props}
      className={`clean-btn tocArrowButton${props.className ? ` ${props.className}` : ''}`}
      title={collapsed ? '展开本页目录' : '折叠本页目录'}
      aria-label={collapsed ? '展开本页目录' : '折叠本页目录'}>
      <span aria-hidden="true" className={`tocArrowIcon${collapsed ? ' is-collapsed' : ''}`}>⌄</span>
    </button>
  );
}
