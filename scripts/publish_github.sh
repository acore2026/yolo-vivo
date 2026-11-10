#!/usr/bin/env bash
# Run from a normal pc-17 terminal; pushes to the existing repository.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
remote_url=git@github.com:acore2026/yolo-vivo.git
[[ -f "$HOME/.ssh/id_rsa" ]] || { echo '未找到 ~/.ssh/id_rsa'; exit 1; }
export GIT_SSH_COMMAND='ssh -i ~/.ssh/id_rsa -o IdentitiesOnly=yes -o ConnectTimeout=15'
# Check access before changing local Git state.
git ls-remote "$remote_url" >/dev/null
git var GIT_AUTHOR_IDENT >/dev/null
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  if [[ -d .git ]] && [[ -n "$(ls -A .git)" ]]; then
    echo '.git 非空但不可识别，请人工检查；脚本未覆盖它。'
    exit 1
  fi
  git init -b main
fi
branch=$(git branch --show-current)
[[ "$branch" == main ]] || { echo "当前分支是 $branch，预期 main；停止以免推错分支。"; exit 1; }
if git remote get-url origin >/dev/null 2>&1; then
  existing=$(git remote get-url origin)
  [[ "$existing" == "$remote_url" || "$existing" == https://github.com/acore2026/yolo-vivo || "$existing" == https://github.com/acore2026/yolo-vivo.git ]] || {
    echo 'origin 指向其他仓库，停止；未覆盖配置。'; exit 1;
  }
else
  git remote add origin "$remote_url"
fi
git remote set-url origin "$remote_url"
git add .gitignore README.md STREAMING.md CLIP-GPU.md service scripts deploy train models
# Git respects both root and train ignore rules. Reject oversized files.
python3 - <<'PY'
import subprocess
from pathlib import Path
files = subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMRT','-z']).decode().split(chr(0))
files = [Path(p) for p in files if p]
big = [str(p) for p in files if p.stat().st_size >= 100*1024*1024]
if big:
    raise SystemExit('GitHub 文件大小超限: ' + ', '.join(big))
print(f'提交文件 {len(files)} 个，总大小 {sum(p.stat().st_size for p in files)/1024/1024:.1f} MiB')
PY
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m 'Add three-flower detection, introductions and streaming service'
fi
git fetch origin
# Preserve a README/license or other existing remote history; never force push.
remote_head=$(git ls-remote --symref origin HEAD | awk '/^ref:/ {print $2}')
if git show-ref --verify --quiet refs/remotes/origin/main; then
  git merge --no-edit --allow-unrelated-histories origin/main
elif [[ -n "$remote_head" ]]; then
  git merge --no-edit --allow-unrelated-histories "origin/${remote_head#refs/heads/}"
fi
git push -u origin main
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo edit acore2026/yolo-vivo --description '识别三类花卉' || echo '代码已推送，介绍更新失败；请手动填写：识别三类花卉'
else
  echo '代码已推送；请在 GitHub 仓库 About 填写介绍：识别三类花卉'
fi
git status --short
git log -1 --format='%h %s'
local_head=$(git rev-parse HEAD)
remote_main=$(git ls-remote origin refs/heads/main | awk '{print $1}')
[[ "$local_head" == "$remote_main" ]] || { echo '远端提交与本地不一致，请检查。'; exit 1; }
echo "已核对远端 main: $remote_main"
