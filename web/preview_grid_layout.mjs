// Shared by generator galleries and Pose Manager; tracks fit the images, not spare cell space.
export function fitPreviewGrid(count, availableW, availableH, aspect, footerHeight = 0) {
    const gap = 8;
    const aspects = Array.from({ length: count }, () => aspect);
    let best = null;
    for (let cols = 1; cols <= count; cols++) {
        const rows = Math.ceil(count / cols);
        const cellW = (availableW - gap * (cols - 1)) / cols;
        const cellH = (availableH - gap * (rows - 1)) / rows;
        if (cellW <= 0 || cellH <= footerHeight) continue;
        const sizes = aspects.map(aspect => {
            const width = Math.min(cellW, (cellH - footerHeight) * aspect);
            return { width, height: width / aspect + footerHeight };
        });
        const areas = sizes.map(size => size.width * size.height);
        const score = Math.min(...areas) * 1000000 + areas.reduce((sum, area) => sum + area, 0);
        if (!best || score > best.score) best = {
            cols, rows, cellW, cellH, sizes, score,
            trackW: Math.max(...sizes.map(size => size.width)),
            trackH: Math.max(...sizes.map(size => size.height)),
        };
    }
    if (best) {
        best.lastRowCount = count % best.cols || best.cols;
        best.lastRowOffset = best.lastRowCount < best.cols ? (best.cols - best.lastRowCount) * (best.trackW + gap) / 2 : 0;
    }
    return best;
}
