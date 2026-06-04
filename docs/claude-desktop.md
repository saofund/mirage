# Drive Mirage from Claude Desktop (MCP)

The `mirage` MCP server exposes the modeling kernel as tools, so Claude (in Desktop,
or any MCP client) can model live: emit op-log commands, read back structured state,
and get **studio renders inline** in the chat.

Verified launch command (this machine): the project venv interpreter is
cwd-independent and has every dependency:

```
D:\dRepo_26\mirage\.venv\Scripts\python.exe -m mirage.mcp_server
```

## 1. Add the server to Claude Desktop

Edit (create if missing) `%APPDATA%\Claude\claude_desktop_config.json`
— i.e. `C:\Users\wuyitao\AppData\Roaming\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mirage": {
      "command": "D:\\dRepo_26\\mirage\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mirage.mcp_server"]
    }
  }
}
```

Portable alternative (if you'd rather not hard-code the venv path, and `uv` is on PATH):

```json
{
  "mcpServers": {
    "mirage": {
      "command": "uv",
      "args": ["run", "--directory", "D:\\dRepo_26\\mirage", "python", "-m", "mirage.mcp_server"]
    }
  }
}
```

## 2. Restart Claude Desktop

The hammer/tools icon should now list the `mirage` tools:
`new_model`, `apply_mesh_op`, `get_mesh_state`, `diagnose_mesh_op`,
`lint_mesh_program`, `render_model`, `render_mesh_marked`, `repair_mesh_geometry`,
`undo_mesh_op`, `get_mesh_program`.

## 3. Ask Claude to model

Natural-language prompts that exercise the whole loop:

- “Use the mirage tools to model a goblet from scratch, then render it.”
- “Start a new model with a cube, inset the top, extrude a boss, then show me a render.”
- “Model a 3-tier wedding cake; after each tier, render and check it’s still a closed manifold.”
- “Tag the top face ‘lid’, subdivide once, and render with the face IDs marked.”

Claude will call `new_model` → `apply_mesh_op` (one op at a time) → `render_model`,
reading `get_mesh_state` between steps. If it emits a bad op (a tag typo, a too-tight
tol, a scalar scale, a numeric string), `apply_mesh_op`'s `auto_repair` fixes it
silently and reports it; intent-changing mistakes come back as ranked `suggestions`
for Claude to choose. `lint` warnings flag silent traps that build cleanly but lose
intent.

> Note: this repo's `.mcp.json` registers the same server for **Claude Code** (via
> `uv run`), so you can also watch it model live in the CLI — run `/mcp` to approve it.
