/**
 * VNCCS Migration Extension
 * Checks for legacy data structure on startup and migrates it to the new location.
 */
(function () {
    const TAG = "[VNCCS Migration]";
    let hasCheckedConfig = false;

    async function checkAndMigrate() {
        if (hasCheckedConfig) return;
        hasCheckedConfig = true; // Ensure we only check once per session load ideally

        console.log(TAG, "Checking for legacy data migration...");
        try {
            const r = await fetch("/vnccs/migrate", { method: "GET" });
            if (!r.ok) {
                console.warn(TAG, "Migration endpoint failed", r.status);
                return;
            }
            const data = await r.json();

            if (data.migrated) {
                const count = data.count || 0;
                const details = (data.details || []).join(", ");
                const msg = `VNCCS Migration: Moved ${count} character(s) from 'VN_CharacterCreatorSuit' to 'VNCCS/Characters'.\nNames: ${details}`;

                console.log(TAG, msg);

                // Show blocking alert or user friendly toast
                if (window.app && window.app.ui && window.app.ui.dialog) {
                    app.ui.dialog.show(msg);
                } else {
                    alert(msg);
                }

                // Hot-patching proved unreliable. Force reload to ensure all nodes pick up the new paths.
                if (confirm("VNCCS Migration Complete!\n\nMoved characters to new folder structure.\nPress OK to reload the page and apply changes.")) {
                    window.location.reload();
                }

            } else if (data.message) {
                console.log(TAG, "No migration needed:", data.message);
            }
        } catch (e) {
            console.error(TAG, "Error checking migration", e);
        }
    }

    // Register extension
    if (window.app && window.app.registerExtension) {
        app.registerExtension({
            name: "vnccs.migration",
            // We run this check when the app loads OR when a VNCCS node is created/configured
            async setup() {
                // Wait a bit for server to be ready
                setTimeout(checkAndMigrate, 2000);
            },
            nodeCreated(node) {
                // Also trigger if user adds a VNCCS node and we haven't checked yet
                // (e.g. if extension loaded before server was fully ready?)
                if (node.comfyClass &&
                    (node.comfyClass.startsWith("CharacterCreator") || node.comfyClass.startsWith("VNCCS"))) {
                    checkAndMigrate();
                }
            }
        });
    } else {
        // Fallback for older environments
        setTimeout(checkAndMigrate, 3000);
    }

})();
