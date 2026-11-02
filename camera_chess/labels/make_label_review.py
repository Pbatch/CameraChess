from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import orjson

from camera_chess.constants import LABELS_DIR

CLASS_NAMES = [
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a before/after HTML label review.")
    parser.add_argument("--report", type=Path, default=Path(LABELS_DIR) / "cleanlab_nyh_report.json")
    parser.add_argument("--backup", type=Path, default=Path(LABELS_DIR) / ".cleanlab/backup")
    parser.add_argument("--dataset", type=Path, default=Path(LABELS_DIR) / "pieces")
    parser.add_argument("--output", type=Path, default=Path(LABELS_DIR) / "cleanlab_label_review.html")
    return parser.parse_args()


def read_labels(path: Path) -> list[list[float]]:
    return [[float(value) for value in line.split()] for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    report = orjson.loads(args.report.read_bytes())
    cards: list[dict[str, Any]] = []
    for record in report["changes"]:
        # Older reports hold paths relative to the labels folder, newer ones absolute paths
        image = Path(LABELS_DIR) / record["image"]
        current_label = Path(LABELS_DIR) / record["label"]
        backup_label = args.backup / current_label.relative_to(args.dataset)
        cards.append(
            {
                "image": Path(os.path.relpath(image, args.output.parent)).as_posix(),
                "name": image.name,
                "score": record["quality_score"],
                "before": read_labels(backup_label),
                "after": read_labels(current_label),
                "changes": record["changes"],
                "types": sorted({change["type"] for change in record["changes"]}),
                "wrong_likelihood": 1 - min(change["confidence"] for change in record["changes"]),
            }
        )

    cards.sort(key=lambda card: (-card["wrong_likelihood"], card["score"], card["name"]))

    data = orjson.dumps({"classes": CLASS_NAMES, "cards": cards}).decode().replace("</", "<\\/")
    html = HTML.replace("__REVIEW_DATA__", data)
    args.output.write_text(html)
    print(f"Wrote {args.output} with {len(cards)} before/after comparisons")


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cleanlab Label Review</title>
  <style>
    :root { color-scheme: dark; --bg: #101412; --panel: #171d19; --line: #2d3831; --text: #eef5ef; --muted: #9aaa9f; --accent: #d7ff5f; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    header { position: sticky; top: 0; z-index: 2; display: flex; align-items: end; justify-content: space-between; gap: 24px; padding: 22px 28px; border-bottom: 1px solid var(--line); background: rgba(16,20,18,.94); backdrop-filter: blur(12px); }
    h1 { margin: 0; font: 700 clamp(22px,4vw,36px)/1 system-ui, sans-serif; letter-spacing: -.04em; }
    .lede { margin-top: 7px; color: var(--muted); }
    .filters { display: flex; gap: 8px; flex-wrap: wrap; }
    button { border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: transparent; color: var(--muted); cursor: pointer; font: inherit; }
    button.active { border-color: var(--accent); background: var(--accent); color: #151a12; }
    main { display: grid; gap: 18px; padding: 22px 28px 60px; }
    article { overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
    .meta { display: flex; justify-content: space-between; gap: 18px; padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .filename { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; }
    .score { flex: none; color: var(--accent); }
    .views { display: grid; grid-template-columns: 1fr 1fr; }
    figure { min-width: 0; margin: 0; padding: 14px; }
    figure + figure { border-left: 1px solid var(--line); }
    figcaption { display: flex; justify-content: space-between; margin-bottom: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 11px; }
    canvas { display: block; width: 100%; height: auto; border-radius: 5px; background: #090b0a; }
    .changes { display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 16px 15px; border-top: 1px solid var(--line); color: var(--muted); }
    .change { padding: 4px 8px; border: 1px solid var(--line); border-radius: 4px; }
    .change b { color: var(--text); }
    .empty { padding: 80px 20px; color: var(--muted); text-align: center; }
    @media (max-width: 760px) { header { align-items: start; flex-direction: column; padding: 18px; } main { padding: 14px; } .views { grid-template-columns: 1fr; } figure + figure { border-left: 0; border-top: 1px solid var(--line); } }
  </style>
</head>
<body>
  <header>
    <div><h1>Label cleanup review</h1><div class="lede"><span id="count"></span> changed images · original vs Cleanlab-assisted correction</div></div>
    <nav class="filters" aria-label="Change filters">
      <button class="active" data-filter="all">All</button><button data-filter="class">Class fixes</button><button data-filter="missing">Added boxes</button>
    </nav>
  </header>
  <main id="review"></main>
  <script>
    const DATA = __REVIEW_DATA__;
    const COLORS = ['#ff6b6b','#ff9f43','#feca57','#ffdf80','#ff7eb6','#c56cf0','#54a0ff','#48dbfb','#1dd1a1','#a8ff78','#f5f6fa','#c8d6e5'];
    const review = document.querySelector('#review');
    const count = document.querySelector('#count');

    function describe(change) {
      if (change.type === 'class') return `<span class="change">class <b>${DATA.classes[change.old_class]}</b> → <b>${DATA.classes[change.new_class]}</b> · ${(change.confidence*100).toFixed(1)}% · IoU ${change.iou.toFixed(3)}</span>`;
      return `<span class="change">added <b>${DATA.classes[change.new_class]}</b> · ${(change.confidence*100).toFixed(1)}%</span>`;
    }

    function draw(canvas, src, labels) {
      const image = new Image();
      image.onload = () => {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        const scale = Math.max(1, image.naturalWidth / 480);
        ctx.lineWidth = 2 * scale;
        ctx.font = `${Math.round(11 * scale)}px ui-monospace, monospace`;
        labels.forEach(([classId, cx, cy, width, height]) => {
          const x = (cx - width / 2) * canvas.width;
          const y = (cy - height / 2) * canvas.height;
          const w = width * canvas.width;
          const h = height * canvas.height;
          const color = COLORS[classId];
          const label = DATA.classes[classId];
          const textWidth = ctx.measureText(label).width + 8 * scale;
          const textHeight = 16 * scale;
          ctx.strokeStyle = color;
          ctx.fillStyle = color;
          ctx.strokeRect(x, y, w, h);
          ctx.fillRect(x, Math.max(0, y - textHeight), textWidth, textHeight);
          ctx.fillStyle = '#101412';
          ctx.fillText(label, x + 4 * scale, Math.max(12 * scale, y - 4 * scale));
        });
      };
      image.src = src;
    }

    function render(filter = 'all') {
      const cards = DATA.cards.filter(card => filter === 'all' || card.types.includes(filter));
      count.textContent = cards.length;
      review.innerHTML = cards.length ? '' : '<div class="empty">No changes match this filter.</div>';
      cards.forEach((card, index) => {
        const article = document.createElement('article');
        article.innerHTML = `<div class="meta"><span class="filename">${card.name}</span><span class="score">new-label risk ${(card.wrong_likelihood*100).toFixed(1)}% · old quality ${card.score.toFixed(4)}</span></div><div class="views"><figure><figcaption><span>Before</span><span>${card.before.length} boxes</span></figcaption><canvas></canvas></figure><figure><figcaption><span>After</span><span>${card.after.length} boxes</span></figcaption><canvas></canvas></figure></div><div class="changes">${card.changes.map(describe).join('')}</div>`;
        review.appendChild(article);
        const canvases = article.querySelectorAll('canvas');
        draw(canvases[0], card.image, card.before);
        draw(canvases[1], card.image, card.after);
      });
    }

    document.querySelectorAll('button[data-filter]').forEach(button => button.addEventListener('click', () => {
      document.querySelectorAll('button[data-filter]').forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      render(button.dataset.filter);
    }));
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
