# fix-parallel-claude-code

Recent Claude Code builds carry a system-prompt line, "Do not call the AgentTool
unless the user requested it", that makes the model stop delegating to subagents
and fall back to slow, serial work (upstream report:
[anthropics/claude-code#80988](https://github.com/anthropics/claude-code/issues/80988)).
This tool automates the second workaround from
[that thread](https://github.com/anthropics/claude-code/issues/80988#issuecomment-5096233443):
a `UserPromptSubmit` hook that reissues, on every turn, a standing request for
delegation on specific classes of work. No binary patching, no dependencies,
one Python file.

## Check if you are affected

Quick one-liner (any line ending in `:2` is an affected build):

```sh
grep -ac "Do not call the AgentTool unless the user requested it" ~/.local/share/claude/versions/*
```

Or let the tool scan the usual locations plus whatever `claude` on your PATH points to:

```sh
uvx --from git+https://github.com/renato-umeton/fix-parallel-claude-code fix-parallel-claude-code check
```

Pass `check --versions-dir DIR` if your versions live elsewhere. Exit codes:
0 clean, 1 at least one version affected, 2 nothing scanned. An absent marker
may also mean the wording changed in that build; the hook is harmless either way.

## Install

```sh
uvx --from git+https://github.com/renato-umeton/fix-parallel-claude-code fix-parallel-claude-code install
```

Without `uv`, clone the repo and run the file directly (Python 3.10+, stdlib only):

```sh
python3 fix_parallel_claude_code.py install
```

Install flags: `--dry-run`, `--message-file PATH` (your own standing-request
text), `--force` (overwrite a differing script). `--claude-dir DIR` is a global
option and must come before the subcommand (`CLAUDE_CONFIG_DIR` works too):

```sh
fix-parallel-claude-code --claude-dir DIR install --dry-run
```

## Verify

```sh
fix-parallel-claude-code status
```

Then restart Claude Code and ask it:

> by default, when I ask anything, will you use AgentTool or not?

Good answer: it will delegate by default for the categories in the standing
request. Bad answer: it will do everything directly because the instructions
forbid the Agent tool.

## Uninstall

```sh
fix-parallel-claude-code uninstall
```

Removes only the hook entry this tool added and deletes the script. Other hooks
and settings are left untouched.

## What it changes

- `~/.claude/hooks/delegation-standing-request.sh`: a small POSIX shell script
  that prints the hook JSON.
- `~/.claude/settings.json`: appends one entry under `hooks.UserPromptSubmit`
  running `sh "$HOME/.claude/hooks/delegation-standing-request.sh"`. All other
  keys are preserved; before any change the file is copied to
  `settings.json.bak-YYYYmmdd-HHMMSS`. The file is rewritten with two-space
  indentation, and hook containers left empty by `uninstall` are pruned.

## Platforms

Tested on macOS. Looking for testers on Windows and Linux (Claude Code on
Windows runs hooks through Git Bash, so `sh` must be available). If you try it,
please open an issue with your OS, Claude Code version, and whether it worked.

## License

MIT
