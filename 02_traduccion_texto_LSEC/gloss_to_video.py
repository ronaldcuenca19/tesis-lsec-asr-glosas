"""Módulo para convertir una secuencia de glosas en un video final en LSEC.

Este archivo fue preparado para notebooks reproducibles en Google Colab.
"""

import os
import re
import json
import unicodedata
import subprocess
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional, Any


# =========================================================
# CONFIGURACIÓN REPRODUCIBLE PARA COLAB
# =========================================================
# Este archivo NO depende de Google Drive ni de rutas personales.
# Se asume que el ZIP de lenguaje_señas ya fue descargado y extraído en:
#
#   /content/lsec_demo/lenguaje_señas/
#
# Estructura esperada:
#
#   lenguaje_señas/
#   ├── a/
#   ├── b/
#   ├── ...
#   ├── z/
#   └── abecedario/
#
# En el notebook solo se debe importar este módulo y llamar a:
#
#   generar_video_desde_glosas("ELLA ESTUDIAR SICOLOGIA")
#

DEFAULT_PROJECT_DIR = Path("/content/lsec_demo")
DEFAULT_SIGNS_ROOT = DEFAULT_PROJECT_DIR / "lenguaje_señas"
DEFAULT_OUTPUT_VIDEO = Path("/content/salida_final.mp4")

# =========================================================
# UTILIDADES GENERALES DE RUTAS
# =========================================================
def pick_first_existing_dir(candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir():
            return p
    return None


def pick_all_existing_dirs(candidates: List[str]) -> List[Path]:
    found = []
    seen = set()
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir():
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                found.append(p)
    return found


def safe_walk_files(root: Path, suffix: str = ".mp4"):
    """
    Recorre archivos sin romperse si hay carpetas inaccesibles.
    """
    if root is None or not root.exists():
        return

    def _onerror(err):
        print(f"[WARN] No se pudo acceder a: {getattr(err, 'filename', 'desconocido')}")

    for dirpath, dirnames, filenames in os.walk(str(root), topdown=True, onerror=_onerror):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in sorted(filenames):
            if fn.lower().endswith(suffix.lower()):
                yield Path(dirpath) / fn


def safe_walk_dirs(root: Path):
    """
    Recorre directorios sin romperse si hay carpetas inaccesibles.
    """
    if root is None or not root.exists():
        return

    def _onerror(err):
        print(f"[WARN] No se pudo acceder a: {getattr(err, 'filename', 'desconocido')}")

    for dirpath, dirnames, _ in os.walk(str(root), topdown=True, onerror=_onerror):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        yield Path(dirpath)
        for d in dirnames:
            yield Path(dirpath) / d


# =========================================================
# UTILIDADES DE TEXTO
# =========================================================
def normalize_token(text: str) -> str:
    text = text.strip()

    # Normalizar primero a forma compuesta
    text = unicodedata.normalize("NFC", text)

    text = text.lower()

    # proteger ñ
    text = text.replace("ñ", "__enie__").replace("Ñ", "__ENIE__")

    # quitar diacríticos del resto
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # restaurar ñ
    text = text.replace("__enie__", "ñ").replace("__ENIE__", "Ñ")

    # quitar signos
    text = re.sub(r"[^\wñÑ0-9]+", "", text, flags=re.UNICODE)
    return text.lower()


def normalize_phrase(text: str) -> str:
    parts = [normalize_token(x) for x in text.strip().split()]
    parts = [p for p in parts if p]
    return " ".join(parts)


def exact_key(text: str) -> str:
    """
    Conserva diferencias como:
    - él
    - el
    - ella
    Solo limpia espacios y hace casefold.
    """
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()


def canonical_gloss_from_alias(alias: str) -> str:
    alias = alias.strip()
    alias = re.sub(r"\(([^\)]*)\)", "", alias)
    alias = re.sub(r"\s+", " ", alias).strip()
    return alias.upper()


def expand_gender_parenthetical(token: str) -> List[str]:
    """
    gato(a) -> gato, gata
    gemelos(as) -> gemelos, gemelas
    """
    token = token.strip()
    m = re.match(r"^(.*?)(\(([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)\))$", token)
    if not m:
        return [token]

    base = m.group(1).strip()
    opt = m.group(3).strip().lower()

    if not base or not opt:
        return [token]

    forms = [base]

    if opt == "a":
        if base.lower().endswith("o"):
            alt = base[:-1] + "a"
        elif base.lower().endswith("e"):
            alt = base[:-1] + "a"
        else:
            alt = base + "a"
        forms.append(alt)

    elif opt == "as":
        if base.lower().endswith("os"):
            alt = base[:-2] + "as"
        elif base.lower().endswith("es"):
            alt = base[:-2] + "as"
        else:
            alt = base + "as"
        forms.append(alt)

    else:
        forms.append(base + opt)

    out = []
    seen = set()
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def expand_parenthetical_phrase(alias: str) -> List[str]:
    """
    enfermo(a) mental -> enfermo mental, enferma mental
    """
    parts = alias.strip().split()
    if not parts:
        return [alias]

    expanded_parts = [expand_gender_parenthetical(p) for p in parts]
    combos = [" ".join(c) for c in product(*expanded_parts)]

    out = []
    seen = set()
    for c in combos:
        c = re.sub(r"\s+", " ", c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def split_gloss_text(gloss_text: str) -> List[str]:
    """
    Divide el texto manual de glosas.
    Conserva DLT(...) como una sola unidad.
    """
    tokens = []
    i = 0
    text = gloss_text.strip()

    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        if text[i:i+4].upper() == "DLT(":
            j = i + 4
            depth = 1
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            tokens.append(text[i:j].strip())
            i = j
        else:
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            tokens.append(text[i:j].strip())
            i = j

    return [t for t in tokens if t]


def longest_phrase_match(
    tokens: List[str],
    start_idx: int,
    exact_index: Dict[str, List[str]],
    norm_index: Dict[str, List[str]],
    max_phrase_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Busca la frase más larga posible desde start_idx.
    Primero exacta, luego normalizada.
    """
    max_len_here = min(max_phrase_len, len(tokens) - start_idx)

    for L in range(max_len_here, 1, -1):
        phrase = " ".join(tokens[start_idx:start_idx + L])

        phrase_exact = exact_key(phrase)
        if phrase_exact in exact_index:
            return {
                "phrase": phrase,
                "norm_phrase": phrase_exact,
                "length": L,
                "path": exact_index[phrase_exact][0],
                "match_mode": "exact_phrase",
            }

        phrase_norm = normalize_phrase(phrase)
        if phrase_norm in norm_index:
            return {
                "phrase": phrase,
                "norm_phrase": phrase_norm,
                "length": L,
                "path": norm_index[phrase_norm][0],
                "match_mode": "normalized_phrase",
            }

    return None


def ffprobe_video_info(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json",
        path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe falló para: {path}\n"
            f"STDERR:\n{result.stderr}"
        )

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])

    if not streams:
        raise ValueError(
            f"No se encontró stream de video en: {path}\n"
            f"Salida ffprobe:\n{result.stdout}"
        )

    video_stream = streams[0]

    width = video_stream.get("width")
    height = video_stream.get("height")

    duration = data.get("format", {}).get("duration")
    if duration is None:
        duration = video_stream.get("duration")

    if width is None or height is None:
        raise ValueError(
            f"El video no tiene width/height detectables: {path}\n"
            f"Salida ffprobe:\n{result.stdout}"
        )

    if duration is None:
        raise ValueError(
            f"No se pudo obtener duración del video: {path}\n"
            f"Salida ffprobe:\n{result.stdout}"
        )

    return {
        "width": int(width),
        "height": int(height),
        "duration": float(duration),
    }


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1


# =========================================================
# RESOLVER GLOSAS A VIDEOS
# =========================================================
class ManualGlossVideoResolver:
    def __init__(
        self,
        signs_root: str,
        alphabet_root: Optional[str] = None,
        extra_sign_roots: Optional[List[str]] = None,
    ) -> None:
        self.signs_root = Path(signs_root)

        self.extra_sign_roots: List[Path] = []
        for p in (extra_sign_roots or []):
            pp = Path(p)
            if pp.exists() and pp.is_dir():
                self.extra_sign_roots.append(pp)

        if alphabet_root is not None:
            self.alphabet_root = Path(alphabet_root)
        else:
            self.alphabet_root = self._auto_find_alphabet_dir()

        self.sign_index_exact: Dict[str, List[str]] = {}
        self.sign_index_norm: Dict[str, List[str]] = {}
        self.alphabet_index: Dict[str, str] = {}
        self.max_phrase_len = 1

        self._build_indices()

    def _all_scan_roots(self) -> List[Path]:
        roots = [self.signs_root] + self.extra_sign_roots
        out = []
        seen = set()

        for r in roots:
            try:
                rr = str(r.resolve())
            except Exception:
                rr = str(r)

            if rr not in seen and r.exists() and r.is_dir():
                seen.add(rr)
                out.append(r)

        return out

    def _auto_find_alphabet_dir(self) -> Optional[Path]:
        # 1) Buscar abecedario dentro de las raíces principales
        candidate_names = {"abecedario"}

        for root in self._all_scan_roots():
            direct = root / "abecedario"
            if direct.exists() and direct.is_dir():
                return direct

            for p in safe_walk_dirs(root):
                if normalize_token(p.name) in candidate_names:
                    return p

        return None

    def _is_under(self, path: Path, root: Optional[Path]) -> bool:
        if root is None:
            return False
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False

    def _add_sign_alias(self, alias: str, path: str) -> None:
        alias_clean = re.sub(r"\s+", " ", alias.strip())
        if not alias_clean:
            return

        key_exact = exact_key(alias_clean)
        key_norm = normalize_phrase(alias_clean)

        self.sign_index_exact.setdefault(key_exact, [])
        if path not in self.sign_index_exact[key_exact]:
            self.sign_index_exact[key_exact].append(path)

        if key_norm:
            self.sign_index_norm.setdefault(key_norm, [])
            if path not in self.sign_index_norm[key_norm]:
                self.sign_index_norm[key_norm].append(path)
            self.max_phrase_len = max(self.max_phrase_len, len(key_norm.split()))

    def _extract_aliases_from_filename(self, stem: str) -> List[str]:
        aliases = [x.strip() for x in stem.split(",") if x.strip()]
        if not aliases:
            aliases = [stem.strip()]
        return aliases

    def _build_sign_index(self) -> None:
        seen_files = set()

        for root in self._all_scan_roots():
            for video_path in safe_walk_files(root, ".mp4"):
                try:
                    real_video = str(video_path.resolve())
                except Exception:
                    real_video = str(video_path)

                if real_video in seen_files:
                    continue
                seen_files.add(real_video)

                if self._is_under(video_path, self.alphabet_root):
                    continue

                stem = video_path.stem
                aliases = self._extract_aliases_from_filename(stem)

                for alias in aliases:
                    expanded = expand_parenthetical_phrase(alias)
                    for item in expanded:
                        self._add_sign_alias(item, str(video_path))

                    canonical = canonical_gloss_from_alias(alias)
                    if canonical:
                        self._add_sign_alias(canonical, str(video_path))

    def _build_alphabet_index(self) -> None:
        """
        Indexa videos del abecedario:
        a.mp4, b.mp4, ..., z.mp4
        """
        if self.alphabet_root is None or not self.alphabet_root.exists():
            print("[WARN] No se encontró carpeta de abecedario.")
            return

        for video_path in safe_walk_files(self.alphabet_root, ".mp4"):
            stem_raw = video_path.stem.strip()
            stem_norm = normalize_token(stem_raw)

            if len(stem_norm) == 1 and stem_norm.isalpha():
                self.alphabet_index[stem_norm] = str(video_path)

    def _build_indices(self) -> None:
        self._build_sign_index()
        self._build_alphabet_index()

    def _extract_dlt_letters(self, gloss: str) -> Optional[List[str]]:
        m = re.match(r"^\s*DLT\((.*?)\)\s*$", gloss, flags=re.IGNORECASE)
        if not m:
            return None

        inside = m.group(1).strip()
        if not inside:
            return []

        parts = inside.split()
        letters = [normalize_token(x) for x in parts if normalize_token(x)]
        return letters

    def resolve_unit(self, unit: str) -> Dict[str, Any]:
        unit = unit.strip()

        # Caso DLT(...)
        dlt_letters = self._extract_dlt_letters(unit)
        if dlt_letters is not None:
            items = []
            for letter in dlt_letters:
                path = self.alphabet_index.get(letter)
                items.append({
                    "source_gloss": unit,
                    "unit": letter,
                    "mode": "letter",
                    "path": path,
                    "found": path is not None,
                })
            return {
                "gloss": unit,
                "mode": "dlt",
                "items": items,
            }

        # Caso exacto sensible
        unit_exact = exact_key(unit)
        if unit_exact in self.sign_index_exact:
            chosen_path = self.sign_index_exact[unit_exact][0]
            return {
                "gloss": unit,
                "mode": "exact_sign",
                "items": [{
                    "source_gloss": unit,
                    "unit": unit,
                    "mode": "sign",
                    "path": chosen_path,
                    "found": True,
                }]
            }

        # Caso normalizado
        norm_unit = normalize_phrase(unit)
        if norm_unit in self.sign_index_norm:
            chosen_path = self.sign_index_norm[norm_unit][0]
            return {
                "gloss": unit,
                "mode": "normalized_sign",
                "items": [{
                    "source_gloss": unit,
                    "unit": norm_unit,
                    "mode": "sign",
                    "path": chosen_path,
                    "found": True,
                }]
            }

        # Fallback a deletreo
        items = []
        for ch in norm_unit:
            if ch == " ":
                continue
            path = self.alphabet_index.get(ch)
            items.append({
                "source_gloss": unit,
                "unit": ch,
                "mode": "letter",
                "path": path,
                "found": path is not None,
            })

        return {
            "gloss": unit,
            "mode": "fallback_dactylology",
            "items": items,
        }

    def gloss_text_to_plan(self, gloss_text: str) -> List[Dict[str, Any]]:
        raw_units = split_gloss_text(gloss_text)
        plan: List[Dict[str, Any]] = []

        i = 0
        while i < len(raw_units):
            unit = raw_units[i]

            # DLT(...) directo
            if re.match(r"^\s*DLT\(", unit, flags=re.IGNORECASE):
                resolved = self.resolve_unit(unit)
                plan.extend(resolved["items"])
                i += 1
                continue

            # frase más larga posible
            match = longest_phrase_match(
                raw_units,
                i,
                self.sign_index_exact,
                self.sign_index_norm,
                self.max_phrase_len,
            )
            if match is not None:
                plan.append({
                    "source_gloss": match["phrase"],
                    "unit": match["phrase"],
                    "mode": "sign",
                    "path": match["path"],
                    "found": True,
                })
                i += match["length"]
                continue

            # unidad sola
            resolved = self.resolve_unit(unit)
            plan.extend(resolved["items"])
            i += 1

        return plan

    def print_plan(self, plan: List[Dict[str, Any]]) -> None:
        print("\n" + "=" * 120)
        print("PLAN DE VIDEOS")
        print("=" * 120)
        for i, item in enumerate(plan, start=1):
            print(f"[{i}] gloss={item['source_gloss']}")
            print(f"    unit = {item['unit']}")
            print(f"    mode = {item['mode']}")
            print(f"    found= {item['found']}")
            print(f"    path = {item['path']}")
        print("=" * 120 + "\n")

    def print_missing(self, plan: List[Dict[str, Any]]) -> None:
        missing = [x for x in plan if not x.get("found")]
        if not missing:
            print("No faltan videos.")
            return

        print("\nFALTAN ESTOS VIDEOS:")
        for item in missing:
            print(f"- gloss={item['source_gloss']} | unit={item['unit']} | mode={item['mode']}")

    def concatenate_plan(
        self,
        plan: List[Dict[str, Any]],
        output_path: str,
        fps: int = 25,
        crf: int = 18,
        preset: str = "veryfast",
        trim_start_non_alpha: float = 0.4,
        trim_end_non_alpha: float = 0.6,
        speed: float = 1.7,
        transition_duration: float = 0.1,
    ) -> str:

        video_paths = [x["path"] for x in plan if x.get("path")]

        if not video_paths:
            raise ValueError("No se encontraron videos para concatenar.")

        # -----------------------------------------------------
        # Obtener información de los videos
        # -----------------------------------------------------
        infos = []

        for path in video_paths:
            info = ffprobe_video_info(path)
            print("OK:", path, info)
            infos.append(info)

        max_w = make_even(max(info["width"] for info in infos))
        max_h = make_even(max(info["height"] for info in infos))

        cmd = ["ffmpeg", "-y"]

        for path in video_paths:
            cmd.extend(["-i", path])

        filter_parts = []
        processed_durations = []

        # -----------------------------------------------------
        # Normalizar cada clip
        # -----------------------------------------------------
        for i, (video_path, info) in enumerate(zip(video_paths, infos)):
            original_duration = info["duration"]
            is_alpha = self._is_under(Path(video_path), self.alphabet_root)

            clip_start = 0.0
            clip_end = original_duration

            filters = []

            # Recortar solamente videos de palabras completas
            if not is_alpha:
                proposed_start = trim_start_non_alpha
                proposed_end = original_duration - trim_end_non_alpha

                # Evitar producir clips demasiado pequeños
                if proposed_end - proposed_start >= 0.25:
                    clip_start = proposed_start
                    clip_end = proposed_end

                    filters.append(
                        f"trim=start={clip_start:.6f}:end={clip_end:.6f}"
                    )

            # Reiniciar marcas temporales después del trim
            filters.append("setpts=PTS-STARTPTS")

            # Ajustar velocidad
            if speed > 0 and speed != 1.0:
                filters.append(f"setpts=PTS/{speed:.6f}")

            # Todos los clips deben tener exactamente las mismas
            # propiedades para que xfade funcione correctamente
            filters.extend([
                f"fps={fps}",
                f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease",
                f"pad={max_w}:{max_h}:(ow-iw)/2:(oh-ih)/2",
                "setsar=1",
                "settb=AVTB",
                "format=yuv420p",
            ])

            filter_parts.append(
                f"[{i}:v]{','.join(filters)}[v{i}]"
            )

            processed_duration = (clip_end - clip_start) / speed
            processed_durations.append(processed_duration)

        # -----------------------------------------------------
        # Crear cadena de transiciones
        # -----------------------------------------------------
        if len(video_paths) == 1:
            filter_parts.append("[v0]null[vout]")

        else:
            shortest_duration = min(processed_durations)

            # La transición nunca debe ocupar una parte grande del clip
            actual_transition = min(
                transition_duration,
                shortest_duration / 4.0,
            )

            # Evitar valores problemáticos
            actual_transition = max(actual_transition, 0.04)

            current_label = "[v0]"
            accumulated_duration = processed_durations[0]

            for i in range(1, len(video_paths)):
                output_label = f"[xf{i}]"

                # El fundido comienza poco antes de terminar
                # el resultado acumulado
                offset = accumulated_duration - actual_transition
                offset = max(offset, 0.0)

                filter_parts.append(
                    f"{current_label}[v{i}]"
                    f"xfade="
                    f"transition=fade:"
                    f"duration={actual_transition:.6f}:"
                    f"offset={offset:.6f}"
                    f"{output_label}"
                )

                current_label = output_label

                accumulated_duration += (
                    processed_durations[i] - actual_transition
                )

            filter_parts.append(
                f"{current_label}format=yuv420p[vout]"
            )

        filter_complex = ";".join(filter_parts)

        print("\nFILTER_COMPLEX:\n")
        print(filter_complex)
        print()

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-an",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

            raise RuntimeError(
                "FFmpeg falló al concatenar los videos."
            )

        return output_path



# =========================================================
# FUNCIÓN PRINCIPAL PARA USAR DESDE UNA CELDA DEL NOTEBOOK
# =========================================================
def generar_video_desde_glosas(
    glosas: str,
    signs_root: str = str(DEFAULT_SIGNS_ROOT),
    output_path: str = str(DEFAULT_OUTPUT_VIDEO),
    alphabet_root: Optional[str] = None,
    extra_sign_roots: Optional[List[str]] = None,
    show_plan: bool = True,
    strict: bool = True,
    fps: int = 25,
    crf: int = 18,
    preset: str = "veryfast",
    trim_start_non_alpha: float = 0.4,
    trim_end_non_alpha: float = 0.6,
    speed: float = 1.7,
    transition_duration: float = 0.1,
) -> str:
    """
    Genera un video final en LSEC a partir de una secuencia de glosas.

    Parámetros:
    - glosas: secuencia de glosas. Ejemplo: "ELLA ESTUDIAR SICOLOGIA".
    - signs_root: carpeta extraída desde el ZIP de lenguaje_señas.
      Por defecto: /content/lsec_demo/lenguaje_señas.
    - output_path: ruta del video final.
    - alphabet_root: carpeta del abecedario. Si no se indica, se usa signs_root/abecedario.
    - extra_sign_roots: carpetas adicionales opcionales.
    - show_plan: muestra el plan de clips encontrados.
    - strict: si True, detiene la generación cuando falta algún clip.
              si False, concatena solo los clips encontrados.
    """

    if not glosas or not glosas.strip():
        raise ValueError("La secuencia de glosas está vacía.")

    signs_root_path = Path(signs_root)

    # Fallback por si el ZIP fue extraído con nombre sin ñ.
    if not signs_root_path.exists():
        alt = DEFAULT_PROJECT_DIR / "lenguaje_senas"
        if alt.exists() and alt.is_dir():
            signs_root_path = alt
        else:
            raise FileNotFoundError(
                f"No se encontró la carpeta de videos de señas: {signs_root_path}\n"
                "Verifica que el ZIP se haya extraído en /content/lsec_demo/."
            )

    if alphabet_root is None:
        alphabet_root_path = signs_root_path / "abecedario"
    else:
        alphabet_root_path = Path(alphabet_root)

    resolver = ManualGlossVideoResolver(
        signs_root=str(signs_root_path),
        alphabet_root=str(alphabet_root_path) if alphabet_root_path.exists() else None,
        extra_sign_roots=extra_sign_roots or [],
    )

    plan = resolver.gloss_text_to_plan(glosas)

    if show_plan:
        resolver.print_plan(plan)
        resolver.print_missing(plan)

        validos = [x["path"] for x in plan if x.get("path")]
        print("Cantidad de clips válidos:", len(validos))
        for i, p in enumerate(validos, 1):
            print(i, p)

    if strict:
        missing = [x for x in plan if not x.get("found")]
        if missing:
            detalle = "\n".join(
                f"- gloss={x['source_gloss']} | unit={x['unit']} | mode={x['mode']}"
                for x in missing
            )
            raise FileNotFoundError(
                "No se puede generar el video porque faltan clips:\n" + detalle
            )

    video_final = resolver.concatenate_plan(
        plan=plan,
        output_path=output_path,
        fps=fps,
        crf=crf,
        preset=preset,
        trim_start_non_alpha=trim_start_non_alpha,
        trim_end_non_alpha=trim_end_non_alpha,
        speed=speed,
        transition_duration=transition_duration,
    )

    print("VIDEO FINAL:", video_final)
    return video_final


if __name__ == "__main__":
    # Prueba rápida opcional si se ejecuta como script.
    generar_video_desde_glosas(
        glosas="ELLA ESTUDIAR SICOLOGIA",
        signs_root=str(DEFAULT_SIGNS_ROOT),
        output_path=str(DEFAULT_OUTPUT_VIDEO),
        show_plan=True,
        strict=False,
    )
