---
name: VNCCS Development Guide
description: Comprehensive guide for developing, refactoring, and maintaining the VNCCS (Visual Novel Character Creator Suite) project.
---

---
name: vnccs-dev
description: Comprehensive guide for developing, refactoring, and maintaining the VNCCS project. Use for architectural decisions and understanding specific VNCCS constants/utils.
---

# VNCCS Development Guide


This skill encapsulates the knowledge required to effectively work with the VNCCS codebase. It covers the architectural structure, coding standards, and common workflows.

## 1. Project Structure Overiew

*   **`nodes/`**: Contains the core Python node implementations.
    *   `vnccs_utils.py`: Home for general-purpose utility nodes (image processing, chroma key, mask extraction, etc.). Keep this file focused on "pure logic" nodes.
    *   `vnccs_qwen_encoder.py`: Specialized node for the VLB/Qwen logic.
    *   `character_creator.py`: Main logic for initializing characters.
    *   `__init__.py`: **CRITICAL**. This file registers all nodes. If you create a new node file, you MUST import its mappings here.
*   **`web/`**: Frontend logic (JavaScript extensions for ComfyUI).
    *   `vnccs_migration.js`: Handles the automatic checking and migration of character data from the old path to the new one.
    *   `vnccs_autofill/`: Contains the logic for the "autofill" widget behavior.
    *   `js/`: General JS utilities.
*   **`utils.py`**: **The Backbone**. Contains core helper functions for file paths, configuration management, and the migration logic itself. **ALL path logic should reside here.**
*   **`__init__.py` (Root)**: Registers API endpoints (backend routes) and exposes the nodes to ComfyUI.

## 2. Key Architectural Decisions

### A. Data Storage & Paths
*   **Old Path (Legacy)**: `output/VN_CharacterCreatorSuit`
*   **New Path (Standard)**: `output/VNCCS/Characters`
*   **Rule**: NEVER hardcode paths. Always use `utils.base_output_dir()` to get the current correct root.
*   **Migration**: The `utils.migrate_legacy_data()` function handles moving data. The `vnccs_migration.js` script triggers this on startup.

### B. Node Registration
Two-step process:
1.  Define `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in your `nodes/my_node.py`.
2.  Import and merge these into `nodes/__init__.py`.
    ```python
    from .my_node import NODE_CLASS_MAPPINGS as MyNodeMappings
    # ... merge logic
    ```

### C. API Endpoints
*   Registered in Root `__init__.py`.
*   Use `aiohttp` via `server.PromptServer`.
*   **Pattern**:
    ```python
    @PromptServer.instance.routes.get("/vnccs/my_endpoint")
    async def my_handler(request):
        # ... logic
        return web.json_response({...})
    ```

## 3. Coding Standards (The "VNCCS Style")

1.  **Imports**:
    *   Avoid circular imports by importing inside functions/methods if necessary (lazy imports).
    *   Use `try/except ImportError` blocks for external dependencies to keep the node loadable even if a specific lib is missing (fallback to graceful error or warning).
2.  **Type Hinting**:
    *   Stronly encouraged for all new code. Use `typing.List`, `typing.Dict`, `typing.Optional`.
    *   Example: `def process(self, image: torch.Tensor, strength: float = 1.0) -> Tuple[torch.Tensor, str]:`
3.  **UI Feedback**:
    *   Use `tooltips` in `INPUT_TYPES` to help users.
    *   Return user-friendly errors in JSON responses for APIs, not just 500s.
4.  **Refactoring**:
    *   When refactoring, check for "Legacy" code markers.
    *   Preserve `print()` statements if they are useful for the user's debugging (as requested by the project owner).

## 4. Common Workflows

### Adding a New Node
1.  Create `nodes/new_node.py`.
2.  Implement class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`.
3.  Add `NODE_CLASS_MAPPINGS`.
4.  Update `nodes/__init__.py` to import it.

### Modifying Character Data Structure
1.  Update `utils.py` helpers (e.g., `save_config`).
2.  Check `web/vnccs_autofill.js` if it relies on a specific JSON schema.
3.  Verify backward compatibility for existing characters in `output/VNCCS/Characters`.

### Debugging Migration
*   Check Server Console for `[VNCCS Migration]` tags.
*   Check Browser Console for `[VNCCS Migration]` tags.
*   Force reload: The `vnccs_migration.js` script will prompt for a reload after migration.

## 5. Known Issues / "Gotchas"
*   **Hot-patching UI**: Updating widgets dynamically without a page reload is unreliable in ComfyUI. Prefer `window.location.reload()` if critical data structures change (like the character list).
*   **Widget Types**: Be careful with custom widgets vs standard Comfy widgets.
