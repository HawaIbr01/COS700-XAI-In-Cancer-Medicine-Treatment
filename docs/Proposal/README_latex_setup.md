# LaTeX Setup in VS Code — Mac and Windows

## Overview

You need two things: a LaTeX distribution (the compiler) and the LaTeX Workshop extension for VS Code (the editor integration). Install the distribution first, then the extension.

---

## Windows

### Step 1 — Install MiKTeX (LaTeX distribution)

1. Go to https://miktex.org/download and download the Windows installer
2. Run the installer — when asked about installing missing packages, select **"Yes" (install on the fly)**
3. After installation, open the **MiKTeX Console** from the Start menu
4. Click **Updates** → **Check for updates** → **Update now**
5. Close MiKTeX Console

### Step 2 — Install Perl (required for latexmk)

LaTeX Workshop uses `latexmk` to build, which requires Perl.

1. Go to https://strawberryperl.com and download the installer
2. Run it with default settings
3. Restart your terminal after installation

### Step 3 — Verify installation

Open a new terminal in VS Code (`Ctrl+`` `) and run:
```
pdflatex --version
latexmk --version
```
Both should print version information. If either says "not recognised", restart VS Code and try again.

### Step 4 — Install LaTeX Workshop extension

1. Open VS Code
2. Press `Ctrl+Shift+X` to open Extensions
3. Search for **LaTeX Workshop** by James Yu
4. Click Install

### Step 5 — Add `.vscode/settings.json`

Copy the `settings.json` from the `.vscode/` folder in this repo into your local `.vscode/` folder. This configures the build recipe correctly for this project. See the settings file for details.

---

## Mac

### Step 1 — Install MacTeX (LaTeX distribution)

**Option A — Full install (recommended, ~4GB, everything included):**
1. Go to https://www.tug.org/mactex/mactex-download.html
2. Download `MacTeX.pkg`
3. Double-click and follow the installer

**Option B — Smaller install with Homebrew (~500MB):**
```bash
brew install --cask basictex
```
Then add the LaTeX binaries to your PATH by adding this line to `~/.zshrc` or `~/.bash_profile`:
```bash
export PATH="/Library/TeX/texbin:$PATH"
```
Then run:
```bash
source ~/.zshrc       # or source ~/.bash_profile
sudo tlmgr update --self
sudo tlmgr install latexmk collection-fontsrecommended
```
BasicTeX is missing some packages by default. If a build fails with "missing package", install it:
```bash
sudo tlmgr install <package-name>
```

### Step 2 — Verify installation

Open a terminal and run:
```bash
pdflatex --version
latexmk --version
```
Both should print version information. If not, check that `/Library/TeX/texbin` is in your PATH.

### Step 3 — Install LaTeX Workshop extension

1. Open VS Code
2. Press `Cmd+Shift+X` to open Extensions
3. Search for **LaTeX Workshop** by James Yu
4. Click Install

### Step 4 — Add `.vscode/settings.json`

Copy the `settings.json` from the `.vscode/` folder in this repo into your local `.vscode/` folder.

---

## Building the Paper

Once setup is complete:

1. Open the repo folder in VS Code (`File → Open Folder`)
2. Open `main.tex`
3. Press `Ctrl+Alt+B` (Windows) or `Cmd+Option+B` (Mac) to build
4. The PDF will open automatically in the side panel

Alternatively click the **green play button** (▶) in the top right when `main.tex` is open.

To clean auxiliary build files (`.aux`, `.log`, `.bbl` etc.):
- Press `Ctrl+Shift+P` / `Cmd+Shift+P` → type **LaTeX Workshop: Clean up auxiliary files**

---

## Troubleshooting

**"I can't find the PDF" / build seems to do nothing**
- Make sure you have `main.tex` open and active when you trigger the build
- Check the LaTeX Workshop output panel: `Ctrl+Shift+P` → LaTeX Workshop: Show LaTeX Output

**"Package X not found" on Windows (MiKTeX)**
- Open MiKTeX Console → Packages → search for the package → Install
- Or let it auto-install when prompted during build

**"Package X not found" on Mac (BasicTeX)**
```bash
sudo tlmgr install <package-name>
```

**Bibliography not showing / citations show as [?]**
- This is normal on first build. LaTeX needs 3 build passes for bibliography to resolve
- LaTeX Workshop handles this automatically via latexmk — just wait for the full build to finish
- If it still shows [?], check that `custom.bib` is in the same folder as `main.tex`

**"File not found" for a section**
- Check that the file exists at the exact path listed in the `\input{}` command in main.tex
- Paths are relative to main.tex — `sections/sec_intro.tex` means the `sections` folder must be next to `main.tex`

**PDF viewer not opening on Mac**
- VS Code uses its own built-in PDF viewer via LaTeX Workshop
- If it does not open automatically, press `Ctrl+Shift+P` → **LaTeX Workshop: View LaTeX PDF**

---

## Useful Keyboard Shortcuts (LaTeX Workshop)

| Action | Windows | Mac |
|--------|---------|-----|
| Build PDF | `Ctrl+Alt+B` | `Cmd+Option+B` |
| View PDF | `Ctrl+Alt+V` | `Cmd+Option+V` |
| Forward search (jump to PDF location) | `Ctrl+Alt+J` | `Cmd+Option+J` |
| Clean auxiliary files | via Command Palette | via Command Palette |
| SyncTeX (click PDF → jump to source) | `Ctrl+click` in PDF | `Cmd+click` in PDF |