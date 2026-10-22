import { readFileSync } from "node:fs";
import vm from "node:vm";

// Exercise sections of the real extension without importing ComfyUI's browser entry point.
export const source = readFileSync(new URL("../../web/vnccs_cg_creator.js", import.meta.url), "utf8");
const commonSource = readFileSync(new URL("../../web/vnccs_common.js", import.meta.url), "utf8");
const transportSource = readFileSync(new URL("../../web/vnccs_transport.js", import.meta.url), "utf8");
export const connectionSource = transportSource.slice(transportSource.indexOf("export function watchConnection")).replace("export ", "");
export const cleanupSource = commonSource.slice(commonSource.indexOf("export function registerCleanup"), commonSource.indexOf("// Each loader owns a guard")).replace("export ", "");
export const syncDOMWidgetWidth = vm.runInThisContext("(" + commonSource.slice(commonSource.indexOf("export function syncDOMWidgetWidth("), commonSource.indexOf("export function syncDOMWidgetWidthSoon(")).replace("export ", "") + ")");
const section = (start, end) => source.slice(source.indexOf(start), end ? source.indexOf(end) : undefined);
export const { BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT } = vm.runInThisContext("(function () {" + section("const BACKGROUND_CREATOR_ZOOM", "const STYLE") + "return { BACKGROUND_CREATOR_ZOOM, LAYOUT_WIDTH, CREATOR_MIN_HEIGHT }; })()");
export const stateSource = section("const MODEL_DEFAULTS", "// Resolve Utils assets");
export const runtimeSource = section("async function loadFactoryRuntime", "class CGQi2Controls");
export const widgetSource = section('const BASE = "/vnccs/cg-creator"', "function hideState");
export const entrySource = section("function hideState");
export const { clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource } =
    vm.runInThisContext("(function () {" + stateSource + "\nreturn { clone, readState, settingsDefaults, frameDimensions, sourceCameraView, MODEL_DEFAULTS, sourceChoices, selectedSource, sameSource }; })()");
