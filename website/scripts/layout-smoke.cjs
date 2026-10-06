/* Compile and run the focused layout smoke test without a test-runner dependency. */
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const dependencyRoots = [path.resolve(__dirname, '..', 'node_modules'), path.resolve(__dirname, '..', 'node_modules', '.pnpm', 'node_modules')];
process.env.NODE_PATH = [process.env.NODE_PATH, ...dependencyRoots].filter(Boolean).join(path.delimiter);
Module._initPaths();
const ts = require('typescript');

const sourceRoot = path.resolve(__dirname, '..', 'src');
const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'btt-layout-smoke-'));
const files = [
  path.join(sourceRoot, 'theme', 'readingLayout.ts'),
  path.join(sourceRoot, 'theme', 'DocRoot', 'Layout', 'index.tsx'),
  path.join(sourceRoot, 'theme', 'DocRoot', 'Layout', 'layout.test.tsx'),
];
const compilerOptions = {
  jsx: ts.JsxEmit.ReactJSX,
  module: ts.ModuleKind.CommonJS,
  target: ts.ScriptTarget.ES2020,
  esModuleInterop: true,
};
for (const file of files) {
  const relative = path.relative(sourceRoot, file).replace(/\.tsx?$/, '.js');
  const destination = path.join(outputRoot, relative);
  fs.mkdirSync(path.dirname(destination), {recursive: true});
  const source = fs.readFileSync(file, 'utf8');
  const result = ts.transpileModule(source, {compilerOptions, fileName: file});
  fs.writeFileSync(destination, result.outputText);
}
fs.copyFileSync(
  path.join(sourceRoot, 'theme', 'DocRoot', 'Layout', 'styles.module.css'),
  path.join(outputRoot, 'theme', 'DocRoot', 'Layout', 'styles.module.css'),
);

const cssClasses = new Proxy({}, {get: (_target, property) => String(property)});
require.extensions['.css'] = (module) => {
  module.exports = cssClasses;
};
const stubDir = fs.mkdtempSync(path.join(os.tmpdir(), 'btt-layout-stubs-'));
const stubs = {
  '@docusaurus/plugin-content-docs/client': 'exports.useDocsSidebar = () => null; exports.useDoc = () => ({toc: [], frontMatter: {}});',
  '@theme/BackToTopButton': 'module.exports = () => null;',
  '@theme/DocRoot/Layout/Sidebar': 'module.exports = () => null;',
  '@theme/DocRoot/Layout/Main': 'module.exports = ({children}) => children;',
  '@theme/DocSidebarItems': 'module.exports = () => null;',
  '@docusaurus/router': 'exports.useLocation = () => ({pathname: "/"});',
};
const resolvedStubs = Object.fromEntries(
  Object.entries(stubs).map(([name, source]) => {
    const file = path.join(stubDir, name.replace(/[^a-z0-9]/gi, '_') + '.cjs');
    fs.writeFileSync(file, source);
    return [name, file];
  }),
);
const resolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  return resolvedStubs[request] || resolve.call(this, request, parent, ...rest);
};
const smoke = require(path.join(outputRoot, 'theme', 'DocRoot', 'Layout', 'layout.test.js'));
smoke.runLayoutSmokeTest();
console.log('layout smoke test passed');
