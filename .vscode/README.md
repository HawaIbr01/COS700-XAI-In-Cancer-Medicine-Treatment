# VS Code Environment Setup

This folder contains shared workspace settings for the project. Any settings that need to be consistent across all team members (e.g. build tools, formatters, extensions) go in `settings.json` here — **not** in your personal user `settings.json`.

**Your personal settings (themes, icon packs, language runtimes like R or Java, font sizes) stay in your own user settings and should never be committed. PLEASE GUYS DON'T DO THIS !!!!!**

---

## LaTeX Setup

The project document is written in LaTeX. Every team member needs a working LaTeX installation and the LaTeX Workshop extension in VS Code.

### Step 1 — Install a LaTeX distribution

**Windows**
1. Download and install [MiKTeX](https://miktex.org/download) — choose the full installation when prompted
2. After installing, open the MiKTeX Console and click **Check for updates** and install all updates
3. MiKTeX will auto-install missing packages on first build — allow this when prompted

**Mac**
1. Download and install [MacTeX](https://www.tug.org/mactex/) — use the full MacTeX package, not BasicTeX
2. This installs everything including `pdflatex` and `bibtex` — no further steps needed
3. Verify installation by opening Terminal and running: `pdflatex --version`

---

### Step 2 — Install the LaTeX Workshop extension in VS Code

1. Open VS Code
2. Go to Extensions (`Ctrl+Shift+X` on Windows, `Cmd+Shift+X` on Mac)
3. Search for **LaTeX Workshop** by James Yu
4. Click Install

The shared `settings.json` in this folder already configures the correct build recipe — you do not need to change any settings manually.

---

### Step 3 — Build the document

1. Open `Proposal/main.tex` in VS Code
2. The LaTeX Workshop extension will activate automatically
3. Press `Ctrl+Alt+B` (Windows) or `Cmd+Option+B` (Mac) to build, **or** click the green play button in the top right of the editor
4. The PDF will open automatically in a VS Code tab

The build runs the full sequence: `pdflatex → bibtex → pdflatex → pdflatex`. This is required to resolve all citations. Do not use the `pdflatex` only recipe — citations will show as `?` in the output.

---

### Troubleshooting

**Citations showing as `?` in the PDF**
Run a clean build: `Ctrl+Shift+P` → `LaTeX Workshop: Clean up auxiliary files`, then rebuild.

**Build fails on first run (Windows/MiKTeX)**
MiKTeX will prompt you to install missing packages — click yes/install for all of them. The first build may take a few minutes.

**`pdflatex: command not found` (Mac)**
Your PATH is not set correctly. Run the following in Terminal and restart VS Code:
```bash
export PATH="/Library/TeX/texbin:$PATH"
```
To make this permanent, add that line to your `~/.zshrc` or `~/.bash_profile`.

**Extension not recognising `.tex` files**
Make sure you opened the **root folder** of the repo in VS Code (`File → Open Folder`), not just an individual file.

---

## Adding Shared Settings

If you need to add a setting that should apply to everyone on the team:

1. Open `.vscode/settings.json`
2. Add your setting there
3. Commit and push — teammates will get it on next pull
4. Use the `chore/` branching convention, example '`chore/added latex compiler settings`'

Do **not** paste personal settings (paths to local software, themes, etc.) into this file.