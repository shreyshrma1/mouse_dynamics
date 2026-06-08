// Mouse tracker — matches pynput CSV format exactly
// Row: [t, t, button, state, x, y]
// Coordinates: screenX/Y (absolute monitor position)

const Tracker = (() => {
    let events = [];
    let sessionStart = null;
    let flushTimer = null;
    let active = false;

    function elapsed() {
        return (Date.now() - sessionStart) / 1000;
    }

    function onMove(e) {
        const t = elapsed();
        events.push([t, t, 'NoButton', 'Move', e.screenX, e.screenY]);
    }

    function onMouseDown(e) {
        const t = elapsed();
        const btn = e.button === 2 ? 'Right' : 'Left';
        events.push([t, t, btn, 'Pressed', e.screenX, e.screenY]);
    }

    function onMouseUp(e) {
        const t = elapsed();
        const btn = e.button === 2 ? 'Right' : 'Left';
        events.push([t, t, btn, 'Released', e.screenX, e.screenY]);
    }

    function onScroll(e) {
        const t = elapsed();
        const dir = e.deltaY > 0 ? 'Down' : 'Up';
        events.push([t, t, 'Scroll', dir, 0, Math.abs(e.deltaY)]);
    }

    async function flush() {
        if (!events.length) return;
        const toSend = events.splice(0);
        try {
            await fetch('/collect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ events: toSend })
            });
        } catch (_) {}
    }

    function start() {
        if (active) return;
        active = true;
        sessionStart = Date.now();
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mouseup', onMouseUp);
        document.addEventListener('wheel', onScroll, { passive: true });
        // flush every 500ms (batch send), periodic disk flush handled server-side
        flushTimer = setInterval(flush, 500);
        window.addEventListener('beforeunload', () => flush());
    }

    function stop() {
        if (!active) return;
        active = false;
        clearInterval(flushTimer);
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mousedown', onMouseDown);
        document.removeEventListener('mouseup', onMouseUp);
        document.removeEventListener('wheel', onScroll);
        flush();
    }

    return { start, stop, flush };
})();