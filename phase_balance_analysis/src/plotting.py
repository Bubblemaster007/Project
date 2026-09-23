"""白底蓝灰工程风格的 PNG、SVG、PDF 图表输出。"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


BLUE = "#2F5D8A"
DARK = "#263746"
MID = "#5B6B78"
GRID = "#DDE5EB"
LIGHT = "#F4F7F9"
WHITE = "#FFFFFF"


@dataclass
class ChartSpec:
    title: str
    y_label: str
    column: str
    value_format: str
    filename: str


SPECS = [
    ChartSpec(
        "电力不足概率",
        "概率/%",
        "electricity_shortage_probability_pct",
        "{:.2f}",
        "01_probability_metrics",
    ),
    ChartSpec(
        "最大电力缺额",
        "功率/MW",
        "maximum_electricity_shortage_mw",
        "{:.2f}",
        "02_power_extreme_metrics",
    ),
    ChartSpec(
        "期望缺供电量",
        "电量/MWh",
        "expected_energy_not_served_mwh",
        "{:.2f}",
        "03_energy_metrics",
    ),
]


def _font_path() -> str:
    """选择支持中文的系统字体。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("未找到支持中文的系统字体")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_path(), size=size)


def _center_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _nice_max(values: list[float]) -> float:
    maximum = max(values) if values else 0.0
    if maximum <= 0:
        return 1.0
    return maximum * 1.18


def _axis_max(spec: ChartSpec, values: list[float]) -> float:
    """概率轴固定为物理上限 100%，其他指标保留合理留白。"""
    if spec.filename == "01_probability_metrics":
        return 100.0
    return _nice_max(values)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    metrics: pd.DataFrame,
    spec: ChartSpec,
    compact: bool = False,
) -> None:
    """在给定区域绘制单指标阶段柱状图。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    title_size = 62 if compact else 96
    label_size = 44 if compact else 62
    value_size = 48 if compact else 62
    tick_size = 42 if compact else 56
    title_font = _font(title_size)
    label_font = _font(label_size)
    value_font = _font(value_size)
    tick_font = _font(tick_size)

    _center_text(draw, ((left + right) / 2, top + title_size / 2), spec.title, title_font, DARK)

    plot_left = left + int(width * 0.13)
    plot_right = right - int(width * 0.04)
    label_band_top = top + title_size + (55 if compact else 75)
    plot_top = label_band_top + value_size + (28 if compact else 35)
    plot_bottom = bottom - (160 if compact else 210)
    plot_height = plot_bottom - plot_top
    values = metrics[spec.column].astype(float).tolist()
    y_max = _axis_max(spec, values)

    for tick in range(6):
        y = plot_bottom - plot_height * tick / 5
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=3 if not compact else 2)
        label = f"{y_max * tick / 5:.1f}"
        text_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (plot_left - (text_box[2] - text_box[0]) - 18, y - (text_box[3] - text_box[1]) / 2),
            label, font=tick_font, fill=MID,
        )
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=DARK, width=4)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=DARK, width=4)
    draw.text((plot_left, label_band_top), spec.y_label, font=tick_font, fill=MID)

    stages = metrics["stage"].astype(str).tolist()
    group_width = (plot_right - plot_left) / max(len(stages), 1)
    bar_width = group_width * (0.34 if compact else 0.30)
    for index, stage in enumerate(stages):
        center = plot_left + group_width * (index + 0.5)
        value = values[index]
        x0 = center - bar_width / 2
        x1 = center + bar_width / 2
        bar_height = (value / y_max) * plot_height
        y0 = plot_bottom - bar_height
        if value > 0:
            draw.rounded_rectangle(
                (x0, y0, x1, plot_bottom),
                radius=6, fill=BLUE,
            )
        else:
            draw.line((x0, plot_bottom - 2, x1, plot_bottom - 2), fill=BLUE, width=4)
        label = spec.value_format.format(value)
        _center_text(
            draw,
            (
                center,
                max(label_band_top + value_size / 2, y0 - value_size),
            ),
            label,
            value_font,
            DARK,
        )
        _center_text(
            draw,
            (center, plot_bottom + (65 if compact else 82)),
            stage,
            label_font,
            DARK,
        )


def _save_pdf_from_image(image: Image.Image, path: Path, dpi: int) -> None:
    rgb = image.convert("RGB")
    rgb.save(path, "PDF", resolution=float(dpi))


def _svg_chart(
    metrics: pd.DataFrame,
    spec: ChartSpec,
    subtitle: str,
    path: Path,
) -> None:
    width, height = 1200, 400
    left, right, top, bottom = 120, 1140, 145, 325
    stages = metrics["stage"].astype(str).tolist()
    values = metrics[spec.column].astype(float).tolist()
    y_max = _axis_max(spec, values)
    font_family = "Microsoft YaHei, PingFang SC, Source Han Sans SC, sans-serif"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="600" y="38" text-anchor="middle" font-family="{font_family}" font-size="34" fill="{DARK}">{escape(spec.title)}</text>',
        f'<text x="600" y="68" text-anchor="middle" font-family="{font_family}" font-size="19" fill="{MID}">{escape(subtitle)}</text>',
        f'<text x="{left}" y="112" font-family="{font_family}" font-size="23" fill="{MID}">{escape(spec.y_label)}</text>',
    ]
    plot_height = bottom - top
    for tick in range(6):
        y = bottom - plot_height * tick / 5
        value = y_max * tick / 5
        parts.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1.2"/>',
            f'<text x="{left - 14}" y="{y + 8:.1f}" text-anchor="end" font-family="{font_family}" font-size="23" fill="{MID}">{value:.1f}</text>',
        ])
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{DARK}" stroke-width="2"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{DARK}" stroke-width="2"/>',
    ])
    group_width = (right - left) / len(stages)
    bar_width = group_width * 0.30
    for index, stage in enumerate(stages):
        center = left + group_width * (index + 0.5)
        value = values[index]
        x = center - bar_width / 2
        height_value = value / y_max * plot_height
        y = bottom - height_value
        if value > 0:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" rx="3" fill="{BLUE}"/>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{max(125, y - 20):.1f}" text-anchor="middle" font-family="{font_family}" font-size="25" fill="{DARK}">{escape(spec.value_format.format(value))}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{bottom + 50}" text-anchor="middle" font-family="{font_family}" font-size="26" fill="{DARK}">{escape(stage)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _draw_individual(
    metrics: pd.DataFrame,
    spec: ChartSpec,
    subtitle: str,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    image = Image.new("RGB", (3600, 1200), WHITE)
    draw = ImageDraw.Draw(image)
    _center_text(draw, (1800, 68), subtitle, _font(54), MID)
    _draw_panel(draw, (120, 110, 3480, 1180), metrics, spec, compact=False)
    png = output_dir / f"{spec.filename}.png"
    svg = output_dir / f"{spec.filename}.svg"
    pdf = output_dir / f"{spec.filename}.pdf"
    image.save(png, "PNG", dpi=(dpi, dpi), optimize=True)
    _save_pdf_from_image(image, pdf, dpi)
    _svg_chart(metrics, spec, subtitle, svg)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf)}


def derive_summary_note(metrics: pd.DataFrame) -> str:
    """依据实际结果生成汇总图底部说明，避免预设结论。"""
    if metrics["maximum_electricity_shortage_mw"].max() <= 1e-9:
        return "本算例四阶段均保持电力平衡，未出现明显电力不足。"
    power_row = metrics.loc[
        metrics["maximum_electricity_shortage_mw"].idxmax()
    ]
    energy_row = metrics.loc[
        metrics["expected_energy_not_served_mwh"].idxmax()
    ]
    recovery = metrics.loc[metrics["stage"] == "灾后恢复"].iloc[0]
    sustained = metrics.loc[metrics["stage"] == "灾害持续"].iloc[0]
    if recovery["expected_energy_not_served_mwh"] < sustained["expected_energy_not_served_mwh"]:
        recovery_text = "灾后恢复阶段累计缺电量较灾害持续阶段下降"
    elif recovery["expected_energy_not_served_mwh"] > sustained["expected_energy_not_served_mwh"]:
        recovery_text = "灾后恢复阶段累计缺电量仍高于灾害持续阶段"
    else:
        recovery_text = "灾后恢复阶段累计缺电量与灾害持续阶段相当"
    return (
        f"{power_row['stage']}出现最大瞬时缺额，"
        f"{energy_row['stage']}的累计缺电量最高；{recovery_text}。"
    )


def _svg_summary(
    metrics: pd.DataFrame,
    subtitle: str,
    note: str,
    path: Path,
) -> None:
    """生成可直接放入 PPT 的 3:1 超宽 SVG 汇总图。"""
    width, height = 1600, 533
    font_family = "Microsoft YaHei, PingFang SC, Source Han Sans SC, sans-serif"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="800" y="43" text-anchor="middle" font-family="{font_family}" font-size="38" fill="{DARK}">极端事件全过程分阶段电力电量平衡指标</text>',
        f'<text x="800" y="72" text-anchor="middle" font-family="{font_family}" font-size="19" fill="{MID}">{escape(subtitle)}</text>',
    ]
    panel_width = 490
    panel_gap = 28
    panel_top, panel_bottom = 84, 425
    for panel_index, spec in enumerate(SPECS):
        x0 = 28 + panel_index * (panel_width + panel_gap)
        x1 = x0 + panel_width
        parts.append(
            f'<rect x="{x0}" y="{panel_top}" width="{panel_width}" height="{panel_bottom-panel_top}" rx="12" fill="{LIGHT}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{(x0+x1)/2}" y="{panel_top+36}" text-anchor="middle" font-family="{font_family}" font-size="26" fill="{DARK}">{escape(spec.title)}</text>'
        )
        values = metrics[spec.column].astype(float).tolist()
        stages = metrics["stage"].astype(str).tolist()
        ymax = _axis_max(spec, values)
        pl, pr = x0 + 68, x1 - 18
        pt, pb = panel_top + 120, panel_bottom - 55
        ph = pb - pt
        parts.extend([
            f'<text x="{pl}" y="{panel_top+83}" font-family="{font_family}" font-size="17" fill="{MID}">{escape(spec.y_label)}</text>',
        ])
        for tick in range(5):
            y = pb - ph * tick / 4
            parts.extend([
                f'<line x1="{pl}" y1="{y:.1f}" x2="{pr}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>',
                f'<text x="{pl-8}" y="{y+6:.1f}" text-anchor="end" font-family="{font_family}" font-size="16" fill="{MID}">{ymax*tick/4:.1f}</text>',
            ])
        group_width = (pr - pl) / 4
        bar_width = group_width * 0.34
        for idx, stage in enumerate(stages):
            center = pl + group_width * (idx + 0.5)
            value = values[idx]
            x = center - bar_width / 2
            bh = value / ymax * ph
            y = pb - bh
            if value > 0:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bh:.1f}" rx="2" fill="{BLUE}"/>'
                )
            parts.append(
                f'<text x="{center:.1f}" y="{max(panel_top+104,y-14):.1f}" text-anchor="middle" font-family="{font_family}" font-size="18" fill="{DARK}">{escape(spec.value_format.format(value))}</text>'
            )
            parts.append(
                f'<text x="{center:.1f}" y="{pb+34}" text-anchor="middle" font-family="{font_family}" font-size="17" fill="{DARK}">{escape(stage)}</text>'
            )
    wrapped = textwrap.wrap(note, width=80)
    for index, line in enumerate(wrapped[:2]):
        parts.append(
            f'<text x="800" y="{466 + index*25}" text-anchor="middle" font-family="{font_family}" font-size="20" fill="{DARK}">{escape(line)}</text>'
        )
    parts.append(
        f'<text x="800" y="520" text-anchor="middle" font-family="{font_family}" font-size="16" fill="{MID}">注：概率按阶段内全部仿真小时统计；电量为每条仿真序列的期望值。</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _draw_summary(
    metrics: pd.DataFrame,
    subtitle: str,
    output_dir: Path,
    dpi: int,
) -> tuple[dict[str, str], str]:
    note = derive_summary_note(metrics)
    image = Image.new("RGB", (4800, 1600), WHITE)
    draw = ImageDraw.Draw(image)
    _center_text(
        draw, (2400, 95),
        "极端事件全过程分阶段电力电量平衡指标",
        _font(108), DARK,
    )
    _center_text(draw, (2400, 185), subtitle, _font(52), MID)
    panel_width, gap = 1480, 90
    for index, spec in enumerate(SPECS):
        left = 70 + index * (panel_width + gap)
        right = left + panel_width
        draw.rounded_rectangle(
            (left, 250, right, 1240),
            radius=30,
            fill=LIGHT,
            outline=GRID,
            width=4,
        )
        short_spec = ChartSpec(
            title=spec.title,
            y_label=spec.y_label,
            column=spec.column,
            value_format=spec.value_format,
            filename=spec.filename,
        )
        _draw_panel(
            draw, (left + 20, 275, right - 20, 1210),
            metrics, short_spec, compact=True,
        )
    wrapped = textwrap.wrap(note, width=80)
    for line_index, line in enumerate(wrapped[:2]):
        _center_text(
            draw, (2400, 1365 + line_index * 60), line, _font(48), DARK
        )
    _center_text(
        draw, (2400, 1515),
        "注：概率按阶段内全部仿真小时统计；电量为每条仿真序列的期望值。",
        _font(40), MID,
    )
    png = output_dir / "04_ppt_summary.png"
    svg = output_dir / "04_ppt_summary.svg"
    pdf = output_dir / "04_ppt_summary.pdf"
    image.save(png, "PNG", dpi=(dpi, dpi), optimize=True)
    _save_pdf_from_image(image, pdf, dpi)
    _svg_summary(metrics, subtitle, note, svg)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf)}, note


def plot_all_metrics(
    metrics: pd.DataFrame,
    figures_dir: Path,
    plotting_config: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], str]:
    """输出三张核心图和一张 3:1 超宽 PPT 汇总图。"""
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(plotting_config.get("dpi", 300))
    subtitle = str(plotting_config.get("figure_label", "测试阶段划分"))
    paths: dict[str, dict[str, str]] = {}
    for spec in SPECS:
        paths[spec.filename] = _draw_individual(
            metrics, spec, subtitle, figures_dir, dpi
        )
    paths["04_ppt_summary"], note = _draw_summary(
        metrics, subtitle, figures_dir, dpi
    )
    return paths, note
