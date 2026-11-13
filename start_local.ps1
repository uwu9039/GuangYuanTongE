# 本地起「校园办事向导」：http://127.0.0.1:7860
#
#   .\start_local.ps1              # 默认开 dense 通道（用 .env 里的百炼 Key）
#   .\start_local.ps1 -NoDense     # 只跑字面通道（断网/省钱/压延迟时用）
#   .\start_local.ps1 -Port 8080   # 换端口
#
# 依赖不在系统 Python 里（本机 pip 在沙箱里装不了），三件事都在仓库内解决：
#   .venv\Lib\site-packages   fastapi / uvicorn / starlette / click（tools\local_install.py 解 wheel 进去的）
#   .tmp\vector               sources\vector 只读，3 个分片在这里拼成 vectors.npy（VECTOR_DIR 指过来）
#   kb.sqlite                 原文/引文/表格行的 FTS 库（tools\build_kb_db.py 建的，398 MB）
param([switch]$NoDense, [int]$Port = 7860)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$env:PYTHONPATH = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
$vectorDir = Join-Path $PSScriptRoot ".tmp\vector"
if (Test-Path (Join-Path $vectorDir "vectors.npy")) {
    $env:VECTOR_DIR = $vectorDir
} else {
    Write-Host "[提示] $vectorDir 里没有 vectors.npy，原文语义检索这条腿会空（问答不受影响）"
}

# dense 默认开：手写改写 149 题的 hit@5 从 88.6% 抬到 99.3%，代价是每个没见过的问句
# 一次云端 embedding（+250~330 ms，命中检索缓存后 ~10 ms）。
if ($NoDense) {
    Remove-Item Env:\RETRIEVAL_DENSE -ErrorAction SilentlyContinue
    Write-Host "dense：关（只跑字面通道）"
} else {
    $env:RETRIEVAL_DENSE = "1"
    Write-Host "dense：开（百炼 qwen3.7-text-embedding）"
}

Write-Host "启动 http://127.0.0.1:$Port  （Ctrl+C 停）"
python -m uvicorn app:app --host 127.0.0.1 --port $Port --workers 1
