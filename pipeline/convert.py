"""步骤 2：格式规范化 —— 把课件变成「逐页图片 + 原生文本」。

工程决策（方案 §3.3）：不为 PPT 和 PDF 各写一套流程，统一收敛到「页码 → 图片」，
这样后面的 AI 解析只需要处理一种输入。

本机（Windows + 桌面版 Office）优先走 Office COM 导出，画质最好且保真；
没有 Office 时依次退回 LibreOffice → 纯文本兜底。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .manifest import ManifestEntry
from .settings import Settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
PDF_SUFFIXES = {".pdf"}
PRESENTATION_SUFFIXES = {".pptx", ".ppt"}
DOCUMENT_SUFFIXES = {".docx", ".doc"}


class ConvertError(RuntimeError):
    """转换失败。消息要说明「试过哪些办法、你该装什么」。"""


@dataclass
class PageAsset:
    page_no: int
    image_path: Path | None
    native_text: str = ""
    has_embedded_image: bool = False


@dataclass
class RenderedDoc:
    doc_key: str
    source_path: Path
    converter: str
    pages: list[PageAsset] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text_only(self) -> bool:
        return all(page.image_path is None for page in self.pages)


# ---------------------------------------------------------------------------
# 图片后处理：控制长边像素，既省 token 又保证公式看得清
# ---------------------------------------------------------------------------

def _normalize_image(path: Path, *, max_pixels: int, quality: int) -> Path:
    with Image.open(path) as image:
        image = image.convert("RGB")
        longest = max(image.size)
        if longest > max_pixels:
            ratio = max_pixels / longest
            new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        target = path.with_suffix(".jpg")
        image.save(target, "JPEG", quality=quality, optimize=True)

    if target != path and path.exists():
        path.unlink()
    return target


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def pdf_to_pages(
    pdf_path: Path,
    out_dir: Path,
    *,
    doc_key: str,
    settings: Settings,
) -> list[PageAsset]:
    try:
        import pymupdf as fitz  # PyMuPDF ≥1.24 的新导入名
    except ImportError:  # pragma: no cover - 旧版本兜底
        import fitz  # type: ignore[no-redef]

    cfg = settings.parsing
    dpi = int(cfg.get("render_dpi", 150))
    max_pixels = int(cfg.get("render_max_pixels", 2000))
    quality = int(cfg.get("jpeg_quality", 88))

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PageAsset] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            image_path = out_dir / f"{doc_key}_page_{index:04d}.jpg"
            pixmap = page.get_pixmap(dpi=dpi)
            pixmap.save(str(image_path))
            image_path = _normalize_image(
                image_path, max_pixels=max_pixels, quality=quality
            )
            pages.append(
                PageAsset(
                    page_no=index,
                    image_path=image_path,
                    native_text=(page.get_text("text") or "").strip(),
                    has_embedded_image=bool(page.get_images(full=True)),
                )
            )

    return pages


# ---------------------------------------------------------------------------
# PPTX / PPT：优先 Office COM，退回 LibreOffice
# ---------------------------------------------------------------------------

def _pptx_native_text(pptx_path: Path) -> list[str]:
    """用 python-pptx 读文本框，作为 hybrid 模式的输入与解析失败时的兜底。"""
    try:
        from pptx import Presentation
    except ImportError:
        return []

    texts: list[str] = []
    try:
        presentation = Presentation(str(pptx_path))
    except Exception:
        return []

    for slide in presentation.slides:
        buffer: list[str] = []
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:  # type: ignore[union-attr]
                line = "".join(run.text for run in paragraph.runs).strip()
                if line:
                    buffer.append(line)
        texts.append("\n".join(buffer))
    return texts


def _export_with_powerpoint(src: Path, out_dir: Path, *, width: int = 1920) -> None:
    """通过 PowerPoint COM 把每一页幻灯片导出为 PNG。"""
    try:
        import win32com.client  # type: ignore
    except ImportError as exc:
        raise ConvertError(
            "未安装 pywin32，无法调用本机 PowerPoint 导出幻灯片。"
            "请执行：pip install pywin32，或安装 LibreOffice 后重试。"
        ) from exc

    app = None
    presentation = None
    attached_to_existing = False

    # 尊重用户已经打开的 PowerPoint：只借用，不退出
    try:
        try:
            app = win32com.client.GetActiveObject("PowerPoint.Application")
            attached_to_existing = True
        except Exception:
            app = win32com.client.Dispatch("PowerPoint.Application")
    except Exception as exc:
        raise ConvertError(
            f"无法启动 PowerPoint 自动化：{exc}\n"
            f"常见原因：未安装 PowerPoint、当前进程不在交互式登录会话里"
            f"（例如在无界面的服务或计划任务中运行），或被安全策略拦截。\n"
            f"最省事的替代办法：手工把 PPT 另存为 PDF，把 PDF 放进 data/raw/ 后重跑。"
        ) from exc

    try:
        if not attached_to_existing:
            try:
                app.Visible = False  # 部分版本不允许隐藏，失败不影响导出
            except Exception:
                pass
        presentation = app.Presentations.Open(
            str(src.resolve()), ReadOnly=True, Untitled=False, WithWindow=False
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        presentation.Export(str(out_dir.resolve()), "PNG", width, int(width * 9 / 16))
    except Exception as exc:
        raise ConvertError(
            f"PowerPoint 导出失败：{exc}\n"
            f"常见原因：文件被占用、受保护视图未放行、或路径含特殊字符。"
            f"可以先把 PPT 手工另存为 PDF，再放进 data/raw/ 重跑。"
        ) from exc
    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if app is not None and not attached_to_existing:
            try:
                app.Quit()
            except Exception:
                pass


def _export_with_libreoffice(src: Path, out_dir: Path) -> None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ConvertError(
            "没找到 LibreOffice（soffice）。若本机没有桌面版 PowerPoint，"
            "请安装 LibreOffice 后重试：https://www.libreoffice.org/download/"
        )

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(src)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        produced = list(Path(tmp).glob("*.pdf"))
        if result.returncode != 0 or not produced:
            raise ConvertError(
                f"LibreOffice 转换失败（退出码 {result.returncode}）："
                f"{(result.stderr or result.stdout or '').strip()[:400]}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced[0], out_dir / "converted.pdf")


def _sort_exported_pngs(directory: Path) -> list[Path]:
    """PowerPoint 导出的文件名带本地化前缀（幻灯片1.PNG / Slide1.PNG），按数字排序。"""
    candidates = [
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ]

    def sort_key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else 10**9, path.name)

    return sorted(candidates, key=sort_key)


def presentation_to_pages(
    src: Path,
    out_dir: Path,
    *,
    doc_key: str,
    settings: Settings,
) -> tuple[list[PageAsset], str]:
    cfg = settings.parsing
    max_pixels = int(cfg.get("render_max_pixels", 2000))
    quality = int(cfg.get("jpeg_quality", 88))

    native_texts = _pptx_native_text(src) if src.suffix.lower() == ".pptx" else []
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / "export"
        try:
            _export_with_powerpoint(src, tmp_dir)
            converter = "powerpoint-com"
        except ConvertError as powerpoint_error:
            try:
                _export_with_libreoffice(src, tmp_dir)
            except ConvertError as libreoffice_error:
                raise ConvertError(
                    f"{src.name}：两条导出路线都失败了。\n"
                    f"— PowerPoint 路线：{powerpoint_error}\n"
                    f"— LibreOffice 路线：{libreoffice_error}"
                ) from libreoffice_error
            pdf_path = tmp_dir / "converted.pdf"
            assets = pdf_to_pages(
                pdf_path, out_dir, doc_key=doc_key, settings=settings
            )
            for asset in assets:
                if asset.page_no <= len(native_texts):
                    asset.native_text = native_texts[asset.page_no - 1]
            return assets, "libreoffice"

        exported = _sort_exported_pngs(tmp_dir)
        if not exported:
            raise ConvertError(
                f"PowerPoint 执行成功但没有导出任何页面，请确认 {src.name} 里确实有幻灯片。"
            )

        pages: list[PageAsset] = []
        for index, png in enumerate(exported, start=1):
            image_path = _normalize_image(png, max_pixels=max_pixels, quality=quality)
            final_path = out_dir / f"{doc_key}_page_{index:04d}.jpg"
            shutil.move(str(image_path), final_path)
            pages.append(
                PageAsset(
                    page_no=index,
                    image_path=final_path,
                    native_text=native_texts[index - 1] if index <= len(native_texts) else "",
                    has_embedded_image=True,
                )
            )

    return pages, converter


# ---------------------------------------------------------------------------
# Word：借 Word 导出 PDF 后复用 PDF 通道
# ---------------------------------------------------------------------------

def _docx_to_pdf(src: Path, work_dir: Path) -> Path:
    try:
        import win32com.client  # type: ignore
    except ImportError as exc:
        raise ConvertError(
            "解析 .docx 需要 pywin32（调用本机 Word 导出 PDF）。"
            "请执行 pip install pywin32，或先把 Word 手工另存为 PDF 放进 data/raw/。"
        ) from exc

    pdf_path = work_dir / f"{src.stem}.pdf"
    app = None
    document = None
    attached_to_existing = False

    try:
        try:
            app = win32com.client.GetActiveObject("Word.Application")
            attached_to_existing = True
        except Exception:
            app = win32com.client.Dispatch("Word.Application")
    except Exception as exc:
        raise ConvertError(
            f"无法启动 Word 自动化：{exc}\n"
            f"最省事的替代办法：手工把文档另存为 PDF，把 PDF 放进 data/raw/ 后重跑。"
        ) from exc

    try:
        if not attached_to_existing:
            try:
                app.Visible = False
            except Exception:
                pass
        document = app.Documents.Open(str(src.resolve()), ReadOnly=True, AddToRecentFiles=False)
        document.SaveAs(str(pdf_path.resolve()), FileFormat=17)  # wdFormatPDF
    except Exception as exc:
        raise ConvertError(
            f"Word 导出 PDF 失败：{exc}\n"
            f"兜底办法：手工在 Word 里另存为 PDF，把 PDF 放进 data/raw/ 后重跑。"
        ) from exc
    finally:
        if document is not None:
            try:
                document.Close(SaveChanges=0)
            except Exception:
                pass
        if app is not None and not attached_to_existing:
            try:
                app.Quit()
            except Exception:
                pass

    if not pdf_path.is_file():
        raise ConvertError(f"Word 报告成功但没有生成 {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def render_document(entry: ManifestEntry, settings: Settings) -> tuple[RenderedDoc, list[str]]:
    """把一份资料转成逐页资源。返回 (结果, 警告列表)。"""
    warnings: list[str] = []
    src = entry.local_path(settings.paths)

    if entry.is_remote:
        raise ConvertError(
            f"{entry.doc_key} 是网络资料（{entry.filename}）。"
            f"三级网络内容需先人工抓取、审核通过后另存为本地文件再登记，"
            f"避免把未经审核的网页直接灌进知识库。"
        )

    if not src.is_file():
        raise ConvertError(
            f"找不到原始文件：{src}\n"
            f"请把《{entry.filename}》放进 {settings.paths.raw_dir}，"
            f"或命名为 {entry.doc_key}{entry.raw_ext or '.pdf'}。"
        )

    suffix = src.suffix.lower()
    work_dir = settings.paths.doc_work(entry.doc_key)
    pages_dir = settings.paths.doc_pages(entry.doc_key)

    if suffix in PDF_SUFFIXES:
        pages = pdf_to_pages(src, pages_dir, doc_key=entry.doc_key, settings=settings)
        return RenderedDoc(entry.doc_key, src, "pymupdf", pages), warnings

    if suffix in PRESENTATION_SUFFIXES:
        pages, converter = presentation_to_pages(
            src, pages_dir, doc_key=entry.doc_key, settings=settings
        )
        if converter == "libreoffice":
            warnings.append(
                f"{entry.doc_key}：未走 Office 导出，已用 LibreOffice 转 PDF 再渲染。"
                f"公式密集的页面建议人工抽查时重点看。"
            )
        return RenderedDoc(entry.doc_key, src, converter, pages), warnings

    if suffix in DOCUMENT_SUFFIXES:
        pdf_path = _docx_to_pdf(src, work_dir)
        pages = pdf_to_pages(pdf_path, pages_dir, doc_key=entry.doc_key, settings=settings)
        return RenderedDoc(entry.doc_key, src, "word-com", pages), warnings

    if suffix in IMAGE_SUFFIXES:
        pages_dir.mkdir(parents=True, exist_ok=True)
        target = pages_dir / f"{entry.doc_key}_page_0001.jpg"
        shutil.copy2(src, target)
        target = _normalize_image(
            target,
            max_pixels=int(settings.parsing.get("render_max_pixels", 2000)),
            quality=int(settings.parsing.get("jpeg_quality", 88)),
        )
        return RenderedDoc(
            entry.doc_key,
            src,
            "image",
            [PageAsset(page_no=1, image_path=target, has_embedded_image=True)],
        ), warnings

    if suffix in TEXT_SUFFIXES:
        text = src.read_text(encoding="utf-8", errors="replace")
        warnings.append(f"{entry.doc_key}：文本类资料，按单页处理，无图片。")
        return RenderedDoc(
            entry.doc_key,
            src,
            "text",
            [PageAsset(page_no=1, image_path=None, native_text=text.strip())],
        ), warnings

    raise ConvertError(
        f"{entry.doc_key} 的格式 {suffix or '（无扩展名）'} 暂不支持。"
        f"支持：PDF、PPTX/PPT、DOCX/DOC、图片、Markdown/TXT。"
    )
