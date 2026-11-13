"""生成评测集文件。

    python -m _eval.build_sets

输出（每行 {"q","gold","source","cat"}）：
    presets.jsonl          763 条人工问法 -> 条目（项目自带）
    negatives.jsonl        A 别的学校 / B 个人数据 / C 库外领域（严格拒答集）
    negatives_hard.jsonl   D 库内有相关条目但没有这一条（难题，单独报告）
    paraphrase.jsonl       手写「另一种说法」集（源文件 paraphrase.tsv）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def main() -> None:
    SETS.mkdir(parents=True, exist_ok=True)
    presets = json.loads((ROOT / "presets.json").read_text(encoding="utf-8"))
    rows = [{"q": q, "gold": [iid], "source": "presets"}
            for iid, qs in presets["coverage"].items() for q in qs]
    (SETS / "presets.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    strict, hard, para = [], [], []
    for ln in (SETS / "negatives_src.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        cat, q = ln.split("\t", 1)
        row = {"q": q.strip(), "gold": [], "source": "negative", "cat": cat}
        (hard if cat == "D" else strict).append(row)
    (SETS / "negatives.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in strict) + "\n", encoding="utf-8")
    (SETS / "negatives_hard.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in hard) + "\n", encoding="utf-8")

    for ln in (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "\t" not in ln:
            continue
        iid, q = ln.split("\t", 1)
        para.append({"q": q.strip(), "gold": [iid.strip()], "source": "paraphrase"})
    (SETS / "paraphrase.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in para) + "\n", encoding="utf-8")

    print(f"presets {len(rows)} / strict-negatives {len(strict)} / hard-negatives {len(hard)} "
          f"/ paraphrase {len(para)}")


if __name__ == "__main__":
    main()
