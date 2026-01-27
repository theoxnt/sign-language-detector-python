# Git Workflow Guide

## 📋 **Version-Controlled Files**

### ✅ **TO COMMIT:**
- Source code (`src/*.py`)
- Tests (`tests/*.py`)
- Configuration (`pyproject.toml`, `.github/`)
- Documentation (`README.md`, `*.md`)
- Templates (`.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`)
- Dockerfile and Docker configuration

### ❌ **DO NOT COMMIT:**
- **Virtual environments**: `venv_*`, `venv_py311/`, `.venv/`
- **User data**: `src/data/Theo/`, `src/data/Yessi/`
- **Generated datasets**: `src/data_pickle/`, `*.pickle`
- **Trained models**: `src/models/*.p`, `*.pth`
- **Build files**: `dist/`, `build/`, `*.egg-info/`
- **Python cache**: `__pycache__/`, `*.pyc`

---

## 🚀 **Recommended Workflow**

### **1. Before committing**

Check for unwanted files:
```bash
git status
```

If you see `venv_*`, `data_pickle`, etc., they are properly ignored (marked `??`).

### **2. Commit only the code**

```bash
# Add CI/CD files
git add .github/ Dockerfile .dockerignore

# Add source code
git add src/*.py tests/*.py

# Add configs
git add pyproject.toml .gitignore README.md

# Commit
git commit -m "ci: Add CI/CD pipeline"
```

### **3. Push to your branch**

```bash
# If you're on theo
git push origin theo

# If you're on yessi
git push origin yessi
```

---

## 🧹 **Cleanup (if files were committed by mistake)**

If you already committed files that should be ignored:

```bash
# Remove from Git cache (keeps files locally)
git rm -r --cached venv_py311 venv_312 src/data_pickle src/models/*.p

# Commit the cleanup
git commit -m "chore: Remove ignored files from git"

# Push
git push origin theo
```

---

## 📦 **Version-Controlled Directory Structure**

```
sign-language-detector-python/
├── .github/                    ✅ VERSIONED
│   ├── workflows/              ✅ CI/CD configs
│   ├── ISSUE_TEMPLATE/         ✅ Templates
│   └── PULL_REQUEST_TEMPLATE.md ✅
├── src/                        ✅ VERSIONED
│   ├── *.py                    ✅ Source code
│   ├── data/                   
│   │   ├── .gitkeep            ✅ Directory structure
│   │   ├── Theo/               ❌ IGNORED (data)
│   │   └── Yessi/              ❌ IGNORED (data)
│   ├── data_pickle/            ❌ IGNORED (datasets)
│   └── models/
│       ├── .gitkeep            ✅ Directory structure
│       └── *.p                 ❌ IGNORED (trained models)
├── tests/                      ✅ VERSIONED
├── venv_py311/                 ❌ IGNORED (environment)
├── venv_312/                   ❌ IGNORED (environment)
├── Dockerfile                  ✅ VERSIONED
├── .dockerignore               ✅ VERSIONED
├── .gitignore                  ✅ VERSIONED
├── pyproject.toml              ✅ VERSIONED
└── README.md                   ✅ VERSIONED
```

---

## ⚠️ **Best Practices**

1. **Never commit**:
   - Your personal virtual environments
   - Your test data
   - Your locally trained models
   - Your local configurations (`.env`, etc.)

2. **Always verify** before pushing:
   ```bash
   git status
   git diff --cached
   ```

3. **`.gitkeep` files**:
   - Allow versioning empty directory structures
   - Contain only comments

4. **If you need to share a model**:
   - Use Git LFS for large files
   - Or share via cloud service
   - Document where to download the model in the README

---

## 🔍 **Check What's Ignored**

```bash
# View all ignored files
git status --ignored

# Test if a file would be ignored
git check-ignore -v venv_py311/
git check-ignore -v src/data_pickle/data_25.pickle
```

---

## 📝 **`.gitkeep` Files**

`.gitkeep` files are placeholders to preserve directory structure in Git:

- `src/data/.gitkeep` - Keeps the data/ directory
- `src/models/.gitkeep` - Keeps the models/ directory

These files allow other developers to clone the project with the correct directory structure.
