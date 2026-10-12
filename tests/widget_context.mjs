import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';

const transport = readFileSync(new URL('../web/vnccs_transport.js', import.meta.url), 'utf8')
    .replace(/^import .*;\n/gm, '').replaceAll('export ', '');
const common = readFileSync(new URL('../web/vnccs_common.js', import.meta.url), 'utf8');
const shared = common.slice(common.indexOf('// ── Shared Input Normalization'), common.indexOf('// ── Debounce'));
const sharedNames = [...shared.matchAll(/export (?:const|function) (\w+)/g)].map(match => match[1]);

// Source-level widget tests still execute the real shared transport helpers.
export function createWidgetContext(sandbox = {}) {
    const api = sandbox.api || {};
    api.apiURL ||= route => route;
    api.fetchApi ||= sandbox.fetch || (() => { throw new Error('Unexpected request'); });
    api.addEventListener ||= () => {};
    api.removeEventListener ||= () => {};
    const context = vm.createContext({
        Headers, crypto: webcrypto, setInterval: () => 0, clearInterval() {},
        window: { addEventListener() {}, removeEventListener() {} },
        document: { visibilityState: 'hidden', addEventListener() {}, removeEventListener() {} },
        ...sandbox, api,
    });
    vm.runInContext(transport, context);
    const helpers = vm.runInContext(`(() => { ${shared.replaceAll('export ', '')}; return { ${sharedNames.join(', ')} }; })()`, context);
    for (const [name, value] of Object.entries(helpers)) {
        if (!(name in sandbox)) context[name] = value;
    }
    return context;
}
