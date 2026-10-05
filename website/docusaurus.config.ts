import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Beneath the Tokens', tagline: 'AI Infrastructure from first principles', favicon: 'img/favicon.ico',
  url: 'https://peppapigw.github.io', baseUrl: '/Beneath-the-Tokens/', organizationName: 'PeppaPigw', projectName: 'Beneath-the-Tokens',
  onBrokenLinks: 'throw', onBrokenMarkdownLinks: 'warn', i18n: {defaultLocale: 'zh-Hans', locales: ['zh-Hans']},
  presets: [['classic', {docs: {path: '../docs', routeBasePath: '/', sidebarPath: './sidebars.ts', showLastUpdateTime: true, breadcrumbs: true, editUrl: 'https://github.com/PeppaPigw/Beneath-the-Tokens/tree/main/'}, blog: false, theme: {customCss: './src/css/custom.css'}} satisfies Preset.Options]],
  themeConfig: {navbar: {title: 'Beneath the Tokens', items: [{type: 'docSidebar', sidebarId: 'bookSidebar', position: 'left', label: '教材'}, {href: 'https://github.com/PeppaPigw/Beneath-the-Tokens', label: 'GitHub'}]}, footer: {style: 'dark', links: [{title: '教材', items: [{label: '课程地图', to: '/curriculum'}]}], copyright: 'Beneath the Tokens'}} satisfies Preset.ThemeConfig,
};
export default config;
