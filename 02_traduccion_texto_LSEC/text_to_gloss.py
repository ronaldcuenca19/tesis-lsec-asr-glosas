import re
import unicodedata
from itertools import product
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from typing import Optional
from threading import Lock

import spacy
import language_tool_python


FUNCTION_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "a", "al", "en", "con", "por", "para",
    "que", "como", "si", "y", "o", "u",
    "haber", "ha", "han", "he", "has",
}

PRESERVE_WORDS = {
    "no", "nunca", "jamas", "jamás",
}

TIME_WORDS = {
    # Referencias generales
    "hoy",
    "ayer",
    "mañana",   
    "manana",   
    "anoche",

    # Partes del día
    "madrugada",
    "mañana",
    "manana",
    "mediodia",
    "tarde",
    "noche",
    "temprano",

    # Referencias relativas
    "ahora",
    "antes",
    "despues",
    "luego",

    # Días
    "lunes",
    "martes",
    "miercoles",
    "jueves",
    "viernes",
    "sabado",
    "domingo",
}


ENCLITIC_PRONOUNS = (
    "nos",
    "les",
    "los",
    "las",
    "me",
    "te",
    "se",
    "lo",
    "la",
    "le",
    "os",
    "mos",
)

PERSONAL_PRONOUN_TO_GLOSS = {
    "yo": "yo",
    "me": "yo",
    "mi": "yo",

    "tu": "tu",
    "te": "tu",
    "ti": "tu",

    "nos": "nosotros",
    "nosotros": "nosotros",
    "nosotras": "nosotros",

    "os": "ustedes",
    "usted": "usted",
    "ustedes": "ustedes",

    "el": "el",
    "lo": "el",
    "le": "el",

    "ella": "ella",
    "la": "ella",

    "ellos": "ellos",
    "los": "ellos",
    "les": "ellos",

    "ellas": "ellas",
    "las": "ellas",
}

ENCLITIC_TO_PRONOUN = {
    "me": "yo",
    "nos": "nosotros",
    "te": "tu",
    "os": "ustedes",
    "lo": "el",
    "la": "ella",
    "los": "ellos",
    "las": "ellas",
    "le": "el",
    "les": "ellos",
    "se": None,
}


RAW_TEMPORAL_MARKER_WORDS = {
    "antes",
    "después", "despues", "depués",
    "luego",
}

PLACE_PREPS = {"a", "al", "en", "hacia"}

PARTIAL_CONJUNCTIONS = {"pero"}

RAW_CONNECTOR_WORDS = {
    "además", "ademas",
    "pero",
    "o",
    "quizá", "quiza",
    "ojalá", "ojala",
}

RAW_CONNECTOR_PHRASES = {
    "por ejemplo",
    "por eso",
    "de pronto",
    "a continuación",
    "a continuacion",
    "tal vez",
}

RAW_PREPOSITION_WORDS = {
    "hasta",
    "desde",
    "sin",
    "sobre",
    "mediante",
    "cada",
}

RAW_PREPOSITION_PHRASES = {
    "a través",
    "a traves",
}

QUESTION_CANONICAL_MAP = {
    "donde": "donde",
    "por que": "porque",
    "por que": "por que",
    "porque": "por que",
    "adonde": "donde",
    "cuando": "cuando",
    "que": "que",
    "quien": "quien",
    "quienes": "quien",
    "cuanto": "cuanto",
    "cuantos": "cuanto",
    "cuanta": "cuanto",
    "cuantas": "cuanto",
    "como": "como",
    "cual": "cual",
    "cuales": "cual",
}

QUESTION_WHERE = {"donde"}
QUESTION_WHEN = {"cuando"}
QUESTION_HOW = {"como"}

BUILTIN_META_GLOSSES = {
    "pasado": "PASADO",
    "futuro": "FUTURO",
}


TIME_UNIT_WORDS = {
    "segundo",
    "minuto",
    "hora",
    "dia",
    "semana",
    "mes",
    "año",
}


EXPLICIT_PRONOUN_FORMS = {
    "yo", "tu", "tú", "el", "él", "ella",
    "nosotros", "nosotras", "ustedes", "ellos", "ellas", "usted",
}

POSSESSIVE_TO_OWNER_PRONOUN = {
    "mi": "mio",
    "mis": "mio",
    "tu": "tú",
    "tus": "tú",
    "nuestro": "nosotros",
    "nuestra": "nosotros",
    "nuestros": "nosotros",
    "nuestras": "nosotros",
    "su": "su",
    "suyo": "suyo",
    "suya": "suya",
}

INFERRED_SUBJECT_MAP = {
    ("1", "Sing"): "yo",
    ("2", "Sing"): "tú",
    ("3", "Sing"): "él",
    ("1", "Plur"): "nosotros",
    ("2", "Plur"): "ustedes",
    ("3", "Plur"): "ellos",
}


# =========================================================
# UTILIDADES
# =========================================================

def normalize_token(text: str) -> str:
    """
    Normalización para matching de glosas:
    - minúsculas
    - quita tildes
    - conserva ñ
    - quita signos
    """
    text = text.lower().strip()
    text = text.replace("ñ", "__enie__").replace("Ñ", "__ENIE__")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("__enie__", "ñ").replace("__ENIE__", "Ñ")
    text = re.sub(r"[^\wñÑ0-9]+", "", text, flags=re.UNICODE)
    return text.lower()


def normalize_phrase(text: str) -> str:
    parts = [normalize_token(x) for x in text.strip().split()]
    parts = [p for p in parts if p]
    return " ".join(parts)


def should_auto_index_output_gloss(gloss: str) -> bool:
    """
    Decide si una glosa de salida también puede indexarse como clave
    de búsqueda en exact_lexicon.

    Evita que glosas funcionales o letras sueltas como "A" se conviertan
    en coincidencias exactas accidentales.
    """
    gloss_norm = normalize_token(gloss)

    if not gloss_norm:
        return False

    function_words_norm = {normalize_token(x) for x in FUNCTION_WORDS}
    if gloss_norm in function_words_norm:
        return False

    if len(gloss_norm) == 1:
        return False

    return True


def spell_gloss(token: str) -> str:
    return "DLT(" + " ".join(list(token.upper())) + ")"


def feats_a_dict(feats_str: str) -> Dict[str, str]:
    if not feats_str:
        return {}
    out = {}
    for item in feats_str.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def clasificar_tiempo_ud(valor_tense: Optional[str]) -> str:
    if valor_tense == "Past":
        return "pasado"
    elif valor_tense == "Imp":
        return "pasado"
    elif valor_tense == "Pres":
        return "presente"
    elif valor_tense == "Fut":
        return "futuro"
    return "desconocido"


def inferir_pronombre(person: Optional[str], number: Optional[str]) -> Optional[str]:
    return INFERRED_SUBJECT_MAP.get((person, number), None)


def canonical_gloss_from_alias(alias: str) -> str:
    alias = alias.strip()
    alias = re.sub(r"\(([^\)]*)\)", "", alias)
    alias = re.sub(r"\s+", " ", alias).strip()
    return alias.upper()


def expand_gender_parenthetical(token: str) -> List[str]:
    """
    Expande una sola palabra:
    - gato(a) -> gato, gata
    - gemelos(as) -> gemelos, gemelas
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
    Expande frases con palabras parentéticas:
    - Enfermo(a) mental -> Enfermo mental, Enferma mental
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


def load_gloss_lexicon(
    txt_path: str
) -> Tuple[
    Dict[str, str],
    Dict[str, List[Dict[str, str]]],
    Dict[str, Dict[str, str]],
    int
]:
    exact_lexicon: Dict[str, str] = {}
    gender_lexicon: Dict[str, List[Dict[str, str]]] = {}
    phrase_lexicon: Dict[str, Dict[str, str]] = {}
    max_phrase_len = 1

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "|" in line:
                alias, gloss = [x.strip() for x in line.split("|", 1)]
                gloss_out = gloss.upper()
            elif "\t" in line:
                alias, gloss = [x.strip() for x in line.split("\t", 1)]
                gloss_out = gloss.upper()
            else:
                alias = line
                gloss_out = canonical_gloss_from_alias(alias)

            alias_clean = re.sub(r"\s+", " ", alias).strip()
            is_phrase = " " in alias_clean

            if is_phrase:
                expanded_phrases = expand_parenthetical_phrase(alias_clean)
                for phr in expanded_phrases:
                    phr_norm = normalize_phrase(phr)
                    if not phr_norm:
                        continue
                    phrase_lexicon[phr_norm] = {
                        "raw_alias": alias_clean,
                        "gloss": gloss_out,
                    }
                    max_phrase_len = max(max_phrase_len, len(phr_norm.split()))
                continue

            expanded_forms = expand_gender_parenthetical(alias_clean)
            is_gender_parenthetical = (
                "(" in alias_clean and ")" in alias_clean and len(expanded_forms) > 1
            )

            if is_gender_parenthetical:
                for form in expanded_forms:
                    form_norm = normalize_token(form)
                    if not form_norm:
                        continue

                    gender_lexicon.setdefault(form_norm, []).append({
                        "raw_alias": alias_clean,
                        "gloss": gloss_out,
                    })

                gloss_norm = normalize_token(gloss_out)
                if (
                    should_auto_index_output_gloss(gloss_out)
                    and gloss_norm not in exact_lexicon
                ):
                    exact_lexicon[gloss_norm] = gloss_out

            else:
                alias_norm = normalize_token(alias_clean)
                if not alias_norm:
                    continue

                exact_lexicon[alias_norm] = gloss_out

                gloss_norm = normalize_token(gloss_out)
                if (
                    should_auto_index_output_gloss(gloss_out)
                    and gloss_norm not in exact_lexicon
                ):
                    exact_lexicon[gloss_norm] = gloss_out

    return exact_lexicon, gender_lexicon, phrase_lexicon, max_phrase_len


def load_manual_synonyms(
    txt_path: str,
    valid_gloss_lexicon: Dict[str, str],
    valid_phrase_lexicon: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Formato aceptado del archivo:

    edad, existencia : año
    realizar : hacer
    ingresar, acceder : pasar
    anoche : ayer, noche
    toros : plaza de toros

    IZQUIERDA:
      - sinónimos/lemmas separados por coma

    DERECHA:
      - si hay comas => se interpreta como SECUENCIA DE GLOSAS
        Ej: ayer, noche  ->  AYER NOCHE
      - si NO hay comas => se interpreta como UN SOLO DESTINO,
        que puede ser:
          a) una palabra que exista en glosses.txt
          b) una frase que exista en glosses.txt como phrase_lexicon
    """
    synonym_map: Dict[str, Dict[str, Any]] = {}

    def resolve_single_right_item(item: str) -> Optional[str]:
        """
        Resuelve un elemento del lado derecho:
        - palabra simple en gloss_lexicon
        - frase en phrase_gloss_lexicon
        """
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            return None

        item_norm_token = normalize_token(item)
        item_norm_phrase = normalize_phrase(item)

        # 1) frase exacta del glosario
        if item_norm_phrase in valid_phrase_lexicon:
            return valid_phrase_lexicon[item_norm_phrase]["gloss"]

        # 2) palabra exacta del glosario
        if item_norm_token in valid_gloss_lexicon:
            return valid_gloss_lexicon[item_norm_token]

        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip()

            if not left or not right:
                continue

            # -------------------------------------------------
            # Resolver la parte derecha
            # -------------------------------------------------
            resolved_output_gloss = None

            # Caso A: secuencia explícita separada por coma
            # Ej: anoche : ayer, noche
            if "," in right:
                right_items = [x.strip() for x in right.split(",") if x.strip()]
                resolved_parts = []

                valid = True
                for item in right_items:
                    resolved = resolve_single_right_item(item)
                    if resolved is None:
                        valid = False
                        break
                    resolved_parts.append(resolved)

                if valid and resolved_parts:
                    resolved_output_gloss = " ".join(resolved_parts)

            # Caso B: un único destino
            # Puede ser palabra o frase
            else:
                resolved_output_gloss = resolve_single_right_item(right)

            if not resolved_output_gloss:
                continue

            # -------------------------------------------------
            # Resolver izquierda: sinónimos/lemmas
            # -------------------------------------------------
            left_items = [x.strip() for x in left.split(",")]

            for item in left_items:
                item_phrase = normalize_phrase(item)
                item_token = normalize_token(item)

                # soporta izquierda simple y también multi-palabra
                left_key = item_phrase if " " in item.strip() else item_token
                if not left_key:
                    continue

                synonym_map[left_key] = {
                    "gloss": resolved_output_gloss,
                    "raw_left": item,
                    "raw_right": right,
                }

    return synonym_map



def unique_nonempty(items: List[str]) -> List[str]:
    out = []
    seen = set()

    for x in items:
        if not x:
            continue
        x = normalize_token(x)
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def participio_a_infinitivos(token: str) -> List[str]:
    """
    Convierte participios/adjetivos verbales en posibles infinitivos.
    No usa tabla manual de irregulares: se apoya primero en el lema de spaCy,
    y solo usa reglas morfológicas como respaldo para casos regulares.
    """
    t = normalize_token(token)
    if not t:
        return []

    candidatos = []

    for suf in ["ados", "adas", "ado", "ada"]:
        if t.endswith(suf) and len(t) > len(suf):
            base = t[:-len(suf)]
            candidatos.append(base + "ar")
            return unique_nonempty(candidatos)

    for suf in ["idos", "idas", "ido", "ida"]:
        if t.endswith(suf) and len(t) > len(suf):
            base = t[:-len(suf)]
            candidatos.append(base + "er")
            candidatos.append(base + "ir")
            return unique_nonempty(candidatos)

    return []


def infinitivo_a_reflexivo(verbo_base: str) -> Optional[str]:
    verbo_base = normalize_token(verbo_base)
    if not verbo_base:
        return None

    if verbo_base.endswith(("ar", "er", "ir")):
        return verbo_base + "se"

    return None


def separar_verbo_y_cliticos(
    verbo: str
) -> Tuple[Optional[str], List[str]]:
    """
    Separa un infinitivo de hasta dos pronombres enclíticos.

    Devuelve:
        (verbo_base, cliticos_en_orden_original)

    Ejemplos:
        visitarme   -> ("visitar", ["me"])
        visitarnos  -> ("visitar", ["nos"])
        conocernos  -> ("conocer", ["nos"])
        decirmelo   -> ("decir", ["me", "lo"])
        darselo     -> ("dar", ["se", "lo"])
        irse        -> ("ir", ["se"])
    """
    actual = normalize_token(verbo)
    if not actual:
        return None, []

    retirados: List[str] = []

    # Hasta dos clíticos: dar + se + lo = dárselo.
    for _ in range(2):
        if retirados and actual.endswith(("ar", "er", "ir")):
            break

        encontrado = None
        for clitico in sorted(ENCLITIC_PRONOUNS, key=len, reverse=True):
            if actual.endswith(clitico) and len(actual) > len(clitico):
                encontrado = clitico
                break

        if encontrado is None:
            break

        actual = actual[:-len(encontrado)]
        retirados.append(encontrado)

    if retirados and actual.endswith(("ar", "er", "ir")):

        return actual, list(reversed(retirados))

    return None, []


def verbo_con_cliticos_a_base(verbo: str) -> Optional[str]:
    """Devuelve solo el infinitivo base de un verbo con clíticos."""
    base, _ = separar_verbo_y_cliticos(verbo)
    return base


def reflexivo_a_base(verbo: str) -> Optional[str]:
    """Compatibilidad con las llamadas antiguas del programa."""
    return verbo_con_cliticos_a_base(verbo)

def limpiar_lema_verbal_spacy(
    texto: str,
    lema_raw: str,
    upos: str,
) -> str:
    """
    Convierte lemas compuestos de spaCy en un lema verbal utilizable.

    Ejemplos observados:
        visitarme  -> "visitar yo"  -> visitar
        conocernos -> "conocer yo"  -> conocer
        ayudarte   -> "ayudar tú"   -> ayudar

    El pronombre del lema de spaCy no se usa para buscar la glosa verbal.
    """
    if upos not in {"VERB", "AUX"}:
        return normalize_token(lema_raw or texto)

    # Es indispensable separar antes de normalizar. Si se normaliza primero,
    # "visitar yo" se convierte en "visitaryo".
    partes = [normalize_token(p) for p in str(lema_raw or "").split()]
    partes = [p for p in partes if p]

    for parte in partes:
        if parte.endswith(("ar", "er", "ir")):
            return parte

    # Respaldo desde la forma superficial.
    base = verbo_con_cliticos_a_base(texto)
    if base:
        return base

    return normalize_token(lema_raw or texto)

def infer_possible_base_verbs(tok: Dict[str, Any]) -> List[str]:
    """
    Recupera infinitivos desde el lema limpio, el lema crudo de spaCy
    y la forma superficial del token.
    """
    surface = normalize_token(tok.get("text", "") or tok.get("norm", ""))
    lemma = normalize_token(tok.get("lemma", ""))
    lemma_raw_text = str(tok.get("lemma_raw", "") or "")
    lemma_raw_parts = [normalize_token(p) for p in lemma_raw_text.split()]
    lemma_raw_parts = [p for p in lemma_raw_parts if p]
    upos = tok.get("upos", "") or ""
    feats = tok.get("feats", "") or ""

    candidatos: List[str] = []

    # Infinitivos ya presentes.
    for cand in [lemma, *lemma_raw_parts, surface]:
        if cand.endswith(("ar", "er", "ir")):
            candidatos.append(cand)

    # Verbos con clíticos: visitarme -> visitar.
    for cand in [lemma, *lemma_raw_parts, surface]:
        base = verbo_con_cliticos_a_base(cand)
        if base:
            candidatos.append(base)

    # Participios regulares como respaldo.
    if "VerbForm=Part" in feats or upos == "ADJ":
        for src in [lemma, *lemma_raw_parts, surface]:
            candidatos.extend(participio_a_infinitivos(src))

    return unique_nonempty(candidatos)

def candidate_lookup_forms(tok: Dict[str, Any]) -> List[str]:
    """
    Prioridad de búsqueda:
    1. forma original
    2. lema
    3. verbo base inferido
    4. forma reflexiva del verbo base
    """
    forms = []

    surface = normalize_token(tok.get("norm", "") or tok.get("text", ""))
    lemma = normalize_token(tok.get("lemma", ""))

    if surface:
        forms.append(surface)

    if lemma and lemma not in forms:
        forms.append(lemma)

    base_verbs = infer_possible_base_verbs(tok)

    for base in base_verbs:
        if base not in forms:
            forms.append(base)

        reflexivo = infinitivo_a_reflexivo(base)
        if reflexivo and reflexivo not in forms:
            forms.append(reflexivo)

    return unique_nonempty(forms)


def has_feat(tok: Dict[str, Any], key: str, value: str) -> bool:
    feats = tok.get("feats", "") or ""
    return f"{key}={value}" in feats


def get_feat_value(tok: Dict[str, Any], key: str) -> Optional[str]:
    feats = tok.get("feats", "") or ""
    parts = feats.split("|") if feats else []
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            if k == key:
                return v
    return None


def is_past_tense(tok: Dict[str, Any]) -> bool:
    return has_feat(tok, "Tense", "Past") or has_feat(tok, "Tense", "Imp")


def is_future_tense(tok: Dict[str, Any]) -> bool:
    return has_feat(tok, "Tense", "Fut")


def is_finite(tok: Dict[str, Any]) -> bool:
    return has_feat(tok, "VerbForm", "Fin")


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    n = len(seq1)
    m = len(seq2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def word_accuracy(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 1.0 if not hyp_words else 0.0

    correct = 0
    total = max(len(ref_words), len(hyp_words))

    min_len = min(len(ref_words), len(hyp_words))
    for i in range(min_len):
        if ref_words[i] == hyp_words[i]:
            correct += 1

    return correct / total if total > 0 else 0.0


# =========================================================
# TRADUCTOR
# =========================================================

class TextToGlossTranslator:
    def __init__(
        self,
        gloss_txt_path: str,
        function_words: set,
        preserve_words: set,
        unknown_mode: str = "spell",
        lt_language: str = "es",
        use_lt_filter: bool = True,
        spacy_model: str = "es_dep_news_trf",
        disable_components: Optional[List[str]] = None,
        manual_synonyms_txt_path: Optional[str] = None,
    ) -> None:
        self.function_words = {normalize_token(x) for x in function_words}
        self.preserve_words = {normalize_token(x) for x in preserve_words}
        self.unknown_mode = unknown_mode
        self.synthetic_uid = -1
        self.use_lt_filter = use_lt_filter

        (
            self.gloss_lexicon,
            self.gender_gloss_lexicon,
            self.phrase_gloss_lexicon,
            self.max_phrase_len
        ) = load_gloss_lexicon(gloss_txt_path)

        self.manual_synonym_map: Dict[str, Dict[str, Any]] = {}
        if manual_synonyms_txt_path:
            self.manual_synonym_map = load_manual_synonyms(
                manual_synonyms_txt_path,
                self.gloss_lexicon,
                self.phrase_gloss_lexicon,
            )

        self.connector_words = {normalize_token(x) for x in RAW_CONNECTOR_WORDS}
        self.connector_phrases = {normalize_phrase(x) for x in RAW_CONNECTOR_PHRASES}
        self.preposition_words = {normalize_token(x) for x in RAW_PREPOSITION_WORDS}
        self.preposition_phrases = {normalize_phrase(x) for x in RAW_PREPOSITION_PHRASES}
        self.temporal_marker_words = {normalize_token(x) for x in RAW_TEMPORAL_MARKER_WORDS}

        if disable_components is None:
            disable_components = ["ner"]

        self.nlp = spacy.load(spacy_model, disable=disable_components)
        self.lt_tool = language_tool_python.LanguageTool(lt_language)

    # -----------------------------------------------------
    # HELPERS
    # -----------------------------------------------------
    def _next_uid(self) -> int:
        self.synthetic_uid -= 1
        return self.synthetic_uid

    def synthetic_candidate(self, token: str, source: str, upos: str = "X") -> Dict[str, Any]:
        norm = normalize_token(token)
        return {
            "uid": self._next_uid(),
            "token": norm,
            "lookup_forms": [norm],
            "surface": norm,
            "lemma_form": norm,
            "upos": upos,
            "source": source,
            "text": token,
        }

    def _match_attr(self, match: Any, *names: str, default=None):
        for name in names:
            if hasattr(match, name):
                return getattr(match, name)
        return default

    def canonical_function_gloss(self, text: str) -> str:
        return normalize_phrase(text).upper().replace(" ", "-")

    def candidate_identity(self, cand: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            cand.get("uid"),
            tuple(cand.get("span_uids", [])),
            tuple(cand.get("lookup_forms", [])),
            cand.get("source"),
        )

    def candidate_position_key(self, cand: Dict[str, Any]) -> float:
        if cand.get("source") in {"negation", "negation_final"}:
            return 1_000_000 + int(
                cand.get("anchor_uid", cand.get("uid", 0)) or 0
            )
        if cand.get("span_uids"):
            return float(min(cand["span_uids"]))

        if isinstance(cand.get("anchor_uid"), int) and cand["anchor_uid"] > 0:
            if cand.get("source") in {
                "subject_inferred",
                "subject_inferred_subordinate",
            }:
                return cand["anchor_uid"] - 0.1

            # Los pronombres enclíticos se colocan inmediatamente
            # después del verbo al que pertenecen.
            if cand.get("source") == "enclitic_object":
                return cand["anchor_uid"] + float(
                    cand.get("anchor_offset", 0.1)
                )

            return float(cand["anchor_uid"])

        uid = cand.get("uid")
        if isinstance(uid, int) and uid > 0:
            return float(uid)

        return 10**9

    def classify_connector_role_from_text(self, text: str) -> str:
        phrase_norm = normalize_phrase(text)
        token_norm = normalize_token(text)

        if (
            phrase_norm in self.attitude_connector_phrases
            or token_norm in self.attitude_connector_words
        ):
            return "attitude"

        if phrase_norm in self.exemplification_connector_phrases:
            return "exemplification"

        if (
            phrase_norm in self.logical_connector_phrases
            or token_norm in self.logical_connector_words
            or token_norm in PARTIAL_CONJUNCTIONS
        ):
            return "logical_bridge"

        return "other"

    def classify_temporal_marker_usage(
        self,
        tok: Dict[str, Any],
        tokens: List[Dict[str, Any]]
    ) -> str:
        norm = tok["norm"]

        if norm not in self.temporal_marker_words:
            return "scene_time"

        has_prev_verb = any(
            t["uid"] < tok["uid"] and t["upos"] in {"VERB", "AUX"}
            for t in tokens
        )
        has_next_verb = any(
            t["uid"] > tok["uid"] and t["upos"] in {"VERB", "AUX"}
            for t in tokens
        )

        if not has_prev_verb:
            return "scene_time"

        if has_prev_verb and has_next_verb:
            return "sequence_time"

        return "scene_time"

    def normalize_time_candidates(self, time_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        meta = []
        other = []

        seen_meta = set()
        seen_other = set()

        for cand in time_candidates:
            tok = cand["token"]

            if tok in BUILTIN_META_GLOSSES:
                if tok not in seen_meta:
                    meta.append(cand)
                    seen_meta.add(tok)
            else:
                key = (tok, cand.get("source", ""))
                if key not in seen_other:
                    other.append(cand)
                    seen_other.add(key)

        return meta + other

    def resolve_gender_parenthetical_candidate(self, token: str, upos: str, source: str) -> Optional[Dict[str, Any]]:
        entries = self.gender_gloss_lexicon.get(token, [])
        if not entries:
            return None

        if upos == "ADJ":
            chosen = entries[0]
            return {
                "input": token,
                "resolved_key": chosen["raw_alias"],
                "gloss": chosen["gloss"],
                "mode": "gender_parenthetical",
                "score": 1.0,
                "source": source,
            }

        if upos in {"NOUN", "PROPN"} and token not in self.gloss_lexicon:
            chosen = entries[0]
            return {
                "input": token,
                "resolved_key": chosen["raw_alias"],
                "gloss": chosen["gloss"],
                "mode": "gender_parenthetical",
                "score": 1.0,
                "source": source,
            }

        return None

    def classify_phrase_role(self, span_tokens: List[Dict[str, Any]]) -> str:
        if not span_tokens:
            return "residual"

        norm_phrase = " ".join(t["norm"] for t in span_tokens if t["norm"])

        if norm_phrase in self.connector_phrases:
            return "conjunctions"

        if norm_phrase in self.preposition_phrases:
            return "prepositions"

        anchor = None
        for tok in span_tokens:
            if tok["upos"] not in {"DET", "ADP", "SCONJ", "CCONJ", "PART", "PUNCT"} and tok["norm"]:
                anchor = tok
                break
        if anchor is None:
            anchor = span_tokens[0]

        first_norm = span_tokens[0]["norm"]

        if first_norm in self.preposition_words:
            return "prepositions"

        if first_norm in PLACE_PREPS:
            return "place"

        if anchor["deprel"] in {"nsubj", "csubj"}:
            return "subject"

        if anchor["deprel"] in {"obj", "iobj"}:
            return "object"

        if anchor["deprel"] == "obl" and first_norm not in self.preposition_words:
            return "place"

        if anchor["upos"] == "ADV":
            return "adverbs"

        if anchor["upos"] == "VERB":
            return "verbs"

        return "residual"

    def build_phrase_candidate(
        self,
        span_tokens: List[Dict[str, Any]],
        entry: Dict[str, str],
        role: str
    ) -> Dict[str, Any]:
        norm_phrase = " ".join(t["norm"] for t in span_tokens if t["norm"])
        text_phrase = " ".join(t["text"] for t in span_tokens)
        upos_anchor = next(
            (t["upos"] for t in span_tokens if t["upos"] not in {"DET", "ADP", "SCONJ", "CCONJ", "PART", "PUNCT"}),
            span_tokens[0]["upos"]
        )

        return {
            "uid": self._next_uid(),
            "token": norm_phrase,
            "lookup_forms": [norm_phrase],
            "surface": norm_phrase,
            "lemma_form": norm_phrase,
            "upos": upos_anchor,
            "source": "phrase",
            "role": role,
            "text": text_phrase,
            "span_uids": [t["uid"] for t in span_tokens],
            "pre_resolved": {
                "input": norm_phrase,
                "resolved_key": entry["raw_alias"],
                "gloss": entry["gloss"],
                "mode": "phrase_exact",
                "score": 1.0,
                "source": "phrase",
            }
        }


    def manual_synonym_candidates(self, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Busca equivalencias únicamente en ``sinonimos.txt``.

        Orden de consulta:
        1. lema producido por spaCy;
        2. forma superficial normalizada.

        No consulta variantes verbales generadas ni WordNet.
        """
        out: List[Dict[str, Any]] = []
        seen_glosses = set()

        lemma_form = normalize_token(candidate.get("lemma_form", ""))
        surface_form = normalize_token(
            candidate.get("surface", "")
            or candidate.get("token", "")
        )

        ordered_forms: List[str] = []
        for form in [lemma_form, surface_form]:
            if form and form not in ordered_forms:
                ordered_forms.append(form)

        for form in ordered_forms:
            entry = self.manual_synonym_map.get(form)
            if not entry:
                continue

            gloss = entry["gloss"]
            if gloss in seen_glosses:
                continue

            seen_glosses.add(gloss)
            out.append({
                "matched_form": form,
                "gloss": gloss,
                "raw_left": entry.get("raw_left"),
                "raw_right": entry.get("raw_right"),
            })

        return out



    def build_builtin_phrase_candidate(
        self,
        span_tokens: List[Dict[str, Any]],
        source: str,
        role: str
    ) -> Dict[str, Any]:
        norm_phrase = " ".join(t["norm"] for t in span_tokens if t["norm"])
        text_phrase = " ".join(t["text"] for t in span_tokens)
        upos_anchor = next(
            (
                t["upos"] for t in span_tokens
                if t["upos"] not in {"DET", "ADP", "SCONJ", "CCONJ", "PART", "PUNCT"}
            ),
            span_tokens[0]["upos"]
        )

        return {
            "uid": self._next_uid(),
            "token": norm_phrase,
            "lookup_forms": [norm_phrase],
            "surface": norm_phrase,
            "lemma_form": norm_phrase,
            "upos": upos_anchor,
            "source": source,
            "role": role,
            "text": text_phrase,
            "span_uids": [t["uid"] for t in span_tokens],
        }

    def collapse_consecutive_resolved_matches(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Colapsa glosas finales consecutivas idénticas.
        Ejemplo:
        IR IR -> IR
        """
        collapsed = []
        last_gloss = None

        for m in matches:
            gloss = (m.get("gloss") or "").strip()

            if collapsed and gloss and gloss == last_gloss:
                continue

            collapsed.append(m)
            last_gloss = gloss

        return collapsed

    # -----------------------------------------------------
    # 0) CORRECCIÓN PREVIA CON LANGUAGETOOL
    # -----------------------------------------------------
    def correct_text(self, text: str) -> Dict[str, Any]:
        matches = self.lt_tool.check(text)

        accepted_matches = matches

        corrected_text = (
            language_tool_python.utils.correct(text, accepted_matches)
            if accepted_matches else text
        )

        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "num_matches_found": len(matches),
            "num_matches_applied": len(accepted_matches),
            "applied_rules": [
                self._match_attr(m, "rule_id", "ruleId", default=None)
                for m in accepted_matches
            ],
        }

    # -----------------------------------------------------
    # 1) ANÁLISIS LINGÜÍSTICO CON SPACY
    # -----------------------------------------------------
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        tokens: List[Dict[str, Any]] = []

        uid = 1
        for sent_idx, sent in enumerate(doc.sents, start=1):
            sent_tokens = [tok for tok in sent if not tok.is_space]
            sent_pos = {tok.i: i for i, tok in enumerate(sent_tokens, start=1)}

            for local_id, tok in enumerate(sent_tokens, start=1):
                feats_str = str(tok.morph)
                feats_dict = tok.morph.to_dict()

                if tok.head == tok:
                    head = 0
                else:
                    head = sent_pos.get(tok.head.i, 0)

                lemma_raw = tok.lemma_ if tok.lemma_ else tok.text
                lemma_norm = limpiar_lema_verbal_spacy(
                    texto=tok.text,
                    lema_raw=lemma_raw,
                    upos=tok.pos_,
                )

                tokens.append({
                    "uid": uid,
                    "sent_id": sent_idx,
                    "id": local_id,
                    "text": tok.text,
                    "norm": normalize_token(tok.text),
                    "lemma": lemma_norm,
                    "lemma_raw": lemma_raw,
                    "upos": tok.pos_,
                    "xpos": tok.tag_,
                    "feats": feats_str,
                    "feats_dict": feats_dict,
                    "head": head,
                    "deprel": tok.dep_.lower(),
                })
                uid += 1

        return tokens

    def detect_sentence_mode(self, raw_text: str, original_tokens: List[Dict[str, Any]]) -> str:
        """
        Una oración solo se considera interrogativa si la constante
        interrogativa aparece al inicio de la oración.
        """
        non_punct_norms = [
            t["norm"]
            for t in original_tokens
            if t["norm"] and t["upos"] != "PUNCT"
        ]

        if not non_punct_norms:
            return "declarative"

        first_norm = non_punct_norms[0]

        if first_norm in QUESTION_CANONICAL_MAP:
            return "interrogative"

        return "declarative"

# =========================================================
# MODIFICA keep_token()
# =========================================================

    def keep_token(self, tok: Dict[str, Any], mode: str) -> bool:
        norm = tok["norm"]
        upos = tok["upos"]
        deprel = tok["deprel"]

        if not norm:
            return False

        if mode == "interrogative" and norm in QUESTION_CANONICAL_MAP:
            return True

        if norm in self.preserve_words:
            return True

        if norm in PARTIAL_CONJUNCTIONS:
            return True

        if norm in self.connector_words or norm in self.preposition_words:
            return True

        if hasattr(self, "temporal_marker_words") and norm in self.temporal_marker_words:
            return True

        # NUEVO: conservar posesivos prenominales
        if norm in POSSESSIVE_TO_OWNER_PRONOUN:
            return True

        if upos == "PRON":
            return True

        if norm in {"el", "ella", "ellos", "ellas", "usted", "ustedes", "yo", "tu", "nosotros", "nosotras"}:
            if deprel in {"nsubj", "csubj"}:
                return True

        if norm in self.function_words and upos in {"DET", "ADP", "SCONJ", "CCONJ", "PART", "AUX"}:
            return False

        if upos == "AUX":
            return False

        if upos in {"DET", "ADP", "SCONJ", "CCONJ", "PART"} and norm not in self.preserve_words:
            return False

        if norm in self.gloss_lexicon or norm in self.gender_gloss_lexicon:
            return True

        return True

    def token_candidate(
        self,
        tok: Dict[str, Any],
        source: str,
        token_override: Optional[str] = None
    ) -> Dict[str, Any]:
        if token_override is not None:
            lookup_forms = [normalize_token(token_override)]
            surface_form = normalize_token(token_override)
            lemma_form = normalize_token(token_override)
        else:
            surface_form = normalize_token(tok.get("norm", "") or tok.get("text", ""))
            lemma_form = normalize_token(tok.get("lemma", ""))
            lookup_forms = candidate_lookup_forms(tok)

            # Protección adicional: derivar la base directamente desde la
            # superficie incluso si el lema de spaCy llega mal formado.
            if tok.get("upos") in {"VERB", "AUX"}:
                direct_base = verbo_con_cliticos_a_base(surface_form)
                if direct_base:
                    lemma_form = direct_base
                    if direct_base not in lookup_forms:
                        lookup_forms.append(direct_base)

                for base in infer_possible_base_verbs(tok):
                    if base not in lookup_forms:
                        lookup_forms.append(base)

        return {
            "uid": tok["uid"],
            # Se conserva la forma superficial como token visible. La búsqueda
            # léxica usa lemma_form y lookup_forms.
            "token": surface_form if surface_form else (lookup_forms[0] if lookup_forms else ""),
            "lookup_forms": lookup_forms,
            "surface": surface_form,
            "lemma_form": lemma_form,
            "upos": tok["upos"],
            "source": source,
            "text": tok["text"],
        }

    def previous_non_punct_token(
        self,
        tokens: List[Dict[str, Any]],
        current_uid: int
    ) -> Optional[Dict[str, Any]]:
        prevs = [t for t in tokens if t["uid"] < current_uid and t["upos"] != "PUNCT"]
        if not prevs:
            return None
        return prevs[-1]

    # -----------------------------------------------------
    # 2) DETECCIÓN DE FRASES DEL GLOSARIO
    # -----------------------------------------------------
    def detect_phrase_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], set]:
        buckets = {
            "subject": [],
            "object": [],
            "place": [],
            "conjunctions": [],
            "prepositions": [],
            "adverbs": [],
            "verbs": [],
            "residual": [],
        }
        all_phrases = []
        used_uids = set()

        seq = [t for t in original_tokens if t["norm"]]

        i = 0
        while i < len(seq):
            if seq[i]["uid"] in blocked_uids or seq[i]["uid"] in used_uids:
                i += 1
                continue

            matched = False
            max_len_here = min(self.max_phrase_len, len(seq) - i)

            for L in range(max_len_here, 1, -1):
                span = seq[i:i + L]
                span_uids = [t["uid"] for t in span]

                if any(uid in blocked_uids or uid in used_uids for uid in span_uids):
                    continue

                norm_phrase = " ".join(t["norm"] for t in span)

                cand = None
                role = None

                # No se aceptan coincidencias exactas de frases del glosario.
                # Solo se conservan frases funcionales internas, como conectores
                # y preposiciones compuestas.
                if norm_phrase in self.connector_phrases:
                    role = "conjunctions"
                    cand = self.build_builtin_phrase_candidate(
                        span_tokens=span,
                        source="conjunction",
                        role=role
                    )
                    cand["connector_role"] = self.classify_connector_role_from_text(norm_phrase)

                elif norm_phrase in self.preposition_phrases:
                    role = "prepositions"
                    cand = self.build_builtin_phrase_candidate(
                        span_tokens=span,
                        source="preposition",
                        role=role
                    )

                if cand is not None:
                    buckets[role].append(cand)
                    all_phrases.append(cand)
                    used_uids.update(span_uids)

                    i += L
                    matched = True
                    break

            if not matched:
                i += 1

        return buckets, all_phrases, used_uids

    # -----------------------------------------------------
    # 3) EXTRACCIÓN DE ESTRUCTURAS
    # -----------------------------------------------------
    def detect_interrogative_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        kept_tokens: List[Dict[str, Any]],
        mode: str
    ) -> List[Dict[str, Any]]:
        out = []

        if mode != "interrogative":
            return out

        seq = [
            t for t in original_tokens
            if t["norm"] and t["upos"] != "PUNCT"
        ]

        if not seq:
            return out

        first_tok = seq[0]
        first_norm = first_tok["norm"]

        if first_norm not in QUESTION_CANONICAL_MAP:
            return out

        # No se busca una frase interrogativa exacta en el glosario.
        # La palabra interrogativa se procesa mediante la regla interna.
        out.append(
            self.token_candidate(
                first_tok,
                source="interrogative",
                token_override=QUESTION_CANONICAL_MAP[first_norm]
            )
        )

        return out



    def detect_possessive_determiner_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> Tuple[List[Dict[str, Any]], set]:
        out = []
        used_uids = set()

        for i, tok in enumerate(original_tokens):
            if tok["uid"] in blocked_uids:
                continue

            norm = tok["norm"]
            if norm not in POSSESSIVE_TO_OWNER_PRONOUN:
                continue

            # Solo un determinante puede funcionar aquí como posesivo.
            # "tu casa" -> DET; "tú vienes" -> PRON.
            if tok.get("upos") != "DET":
                continue

            owner_pron = POSSESSIVE_TO_OWNER_PRONOUN[norm]

            # buscar el sustantivo cercano a la derecha
            j = i + 1
            noun_tok = None
            while j < len(original_tokens):
                nxt = original_tokens[j]

                if nxt["uid"] in blocked_uids:
                    break

                if nxt["upos"] in {"ADJ", "DET", "NUM"}:
                    j += 1
                    continue

                if nxt["upos"] in {"NOUN", "PROPN"}:
                    noun_tok = nxt
                break

            if noun_tok is None:
                continue

            owner_cand = self.synthetic_candidate(
                owner_pron,
                source="possessive_owner",
                upos="PRON"
            )
            owner_cand["anchor_uid"] = tok["uid"]
            out.append(owner_cand)

            # 2) marcar el posesivo como usado
            used_uids.add(tok["uid"])

            # 3) opcional: también marcar el sustantivo para no duplicar raro
            # NO lo bloqueamos aquí, porque "carro" sí debe seguir apareciendo

        return out, used_uids



    def detect_time_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> List[Dict[str, Any]]:
        out = []

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            norm = tok["norm"]

            if norm in TIME_WORDS or norm in self.temporal_marker_words:
                prev_tok = self.previous_non_punct_token(kept_tokens, tok["uid"])

                if prev_tok and prev_tok["norm"] in self.preposition_words:
                    continue

                cand = self.token_candidate(tok, source="time")
                cand["time_role"] = self.classify_temporal_marker_usage(tok, kept_tokens)
                out.append(cand)

        return out




    def interrogative_is_subject(
        self,
        tokens: List[Dict[str, Any]]
    ) -> bool:
        """
        Determina si QUIÉN/QUIÉNES funciona como sujeto de la oración.

        Ejemplo:
            ¿Quién llegó a casa?
            quien -> nsubj de llegó

        En ese caso no se debe inferir otro sujeto como ÉL.
        """
        for tok in tokens:
            norm = tok.get("norm", "")
            deprel = tok.get("deprel", "")

            if (
                norm in {"quien", "quienes"}
                and deprel in {"nsubj", "csubj", "nsubj:pass"}
            ):
                return True

        return False




    def detect_negation_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set,
    ) -> List[Dict[str, Any]]:
        """
        Detecta palabras de negación y las conserva literalmente.

        Ejemplos:
            no     -> NO
            nunca  -> NUNCA
            jamás  -> JAMAS

        Además, se marca la negación con source='negation_final'
        para poder ubicarla al final de la oración.
        """
        out: List[Dict[str, Any]] = []

        negation_map = {
            "no": "no",
            "nunca": "nunca",
            "jamas": "jamas",
            "jamás": "jamas",
        }

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            norm = tok.get("norm", "")

            if norm not in negation_map:
                continue

            literal_negation = negation_map[norm]

            cand = self.token_candidate(
                tok,
                source="negation_final",
                token_override=literal_negation,
            )

            # Se usa para ordenar al final si tu código ordena por posición.
            cand["anchor_uid"] = tok["uid"]

            out.append(cand)

        return out

    def detect_possession_candidates(self, tokens: List[Dict[str, Any]], blocked_uids: set) -> Tuple[List[Dict[str, Any]], set]:
        out: List[Dict[str, Any]] = []
        used_uids = set()
        nounish = {"NOUN", "PROPN", "PRON"}

        i = 0
        while i < len(tokens) - 2:
            left = tokens[i]

            if left["uid"] in blocked_uids:
                i += 1
                continue

            if left["upos"] not in nounish:
                i += 1
                continue

            mid = tokens[i + 1]
            if mid["uid"] in blocked_uids or mid["norm"] not in {"de", "del"}:
                i += 1
                continue

            j = i + 2
            span_uids = {left["uid"], mid["uid"]}

            while j < len(tokens) and tokens[j]["upos"] in {"DET", "ADJ", "NUM"}:
                if tokens[j]["uid"] in blocked_uids:
                    break
                span_uids.add(tokens[j]["uid"])
                j += 1

            if j < len(tokens) and tokens[j]["upos"] in nounish and tokens[j]["uid"] not in blocked_uids:
                right = tokens[j]
                span_uids.add(right["uid"])

                out.append(self.token_candidate(right, source="possession"))
                out.append(self.token_candidate(left, source="possession"))

                used_uids.update(span_uids)
                i = j + 1
            else:
                i += 1

        return out, used_uids

    def detect_place_candidates(self, tokens: List[Dict[str, Any]], blocked_uids: set) -> Tuple[List[Dict[str, Any]], set]:
        out: List[Dict[str, Any]] = []
        used_uids = set()
        nounish = {"NOUN", "PROPN", "PRON"}

        i = 0
        while i < len(tokens) - 1:
            cur = tokens[i]

            if cur["uid"] in blocked_uids or cur["norm"] not in PLACE_PREPS:
                i += 1
                continue

            if cur["norm"] in self.preposition_words:
                i += 1
                continue

            j = i + 1
            span_uids = {cur["uid"]}

            while j < len(tokens) and tokens[j]["upos"] in {"DET", "ADJ", "NUM"}:
                if tokens[j]["uid"] in blocked_uids:
                    break
                span_uids.add(tokens[j]["uid"])
                j += 1

            if j < len(tokens) and tokens[j]["upos"] in nounish and tokens[j]["uid"] not in blocked_uids:
                noun_tok = tokens[j]
                span_uids.add(noun_tok["uid"])

                out.append(self.token_candidate(noun_tok, source="place"))
                used_uids.update(span_uids)
                i = j + 1
            else:
                i += 1

        return out, used_uids


    def infer_subject_from_verb(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set,
    ) -> List[Dict[str, Any]]:
        """
        Infiere un pronombre sujeto a partir de la persona y número
        del verbo finito.

        Además, si encuentra un clítico pronominal redundante antes
        del verbo y coincide con el sujeto inferido, lo marca para ser
        consumido.

        Ejemplo:
            Cuando nos iremos de vacaciones
            -> PREGUNTA CUANDO NOSOTROS IR VACACION

        En este caso:
            iremos -> NOSOTROS
            nos    -> se consume, porque pertenece a 'nos iremos'
        """

        interrogative_subject_forms = {
            "quien",
            "quienes",
            "que",
            "cual",
            "cuales",
            "cuanto",
            "cuantos",
            "cuanta",
            "cuantas",
        }

        for tok in original_tokens:
            norm = normalize_token(
                tok.get("norm", "") or tok.get("text", "")
            )

            deprel = (
                tok.get("deprel", "") or ""
            ).lower()

            if (
                norm in interrogative_subject_forms
                and deprel in {
                    "nsubj",
                    "csubj",
                    "nsubj:pass",
                }
            ):
                return []

        candidatos = []

        for tok in original_tokens:
            if tok["uid"] in blocked_uids:
                continue

            if tok["upos"] not in {"VERB", "AUX"}:
                continue

            feats = tok.get("feats_dict", {})

            if feats.get("VerbForm") != "Fin":
                continue

            if tok.get("lemma") in {"ser", "haber"}:
                continue

            if self.verb_has_explicit_subject(tok, original_tokens):
                continue

            pron = self.infer_pronoun_from_verb_features(tok)

            if pron is None:
                continue

            candidatos.append({
                "tok": tok,
                "pron": pron,
                "priority": 0 if tok["deprel"] == "root" else 1,
            })

        if not candidatos:
            return []

        candidatos.sort(
            key=lambda x: x["priority"]
        )

        elegido = candidatos[0]
        verb_tok = elegido["tok"]
        inferred_pron = elegido["pron"]

        cand = self.synthetic_candidate(
            inferred_pron,
            source="subject_inferred",
            upos="PRON",
        )

        cand["anchor_uid"] = verb_tok["uid"]

        linked_pronoun_uids = set()

        clitic_to_pronoun = {
            "me": "yo",
            "te": "tu",
            "nos": "nosotros",
            "os": "ustedes",
        }

        verb_uid = verb_tok.get("uid", 0)
        sent_id = verb_tok.get("sent_id")

        for tok in original_tokens:
            if tok.get("sent_id") != sent_id:
                continue

            tok_uid = tok.get("uid", 0)
            norm = tok.get("norm", "")

            if norm not in clitic_to_pronoun:
                continue

            # Debe aparecer antes y cerca del verbo.
            if not (0 < verb_uid - tok_uid <= 3):
                continue

            # Solo se consume si representa el mismo sujeto inferido.
            if clitic_to_pronoun[norm] != inferred_pron:
                continue

            linked_pronoun_uids.add(tok_uid)

        cand["linked_pronoun_uids"] = linked_pronoun_uids

        return [cand]



    def verb_has_explicit_subject(
        self,
        verb_tok: Dict[str, Any],
        original_tokens: List[Dict[str, Any]],
    ) -> bool:
        """
        Devuelve True si el verbo ya tiene sujeto explícito.

        Ejemplo:
            las clases de matemáticas comienzan mañana

        En ese caso, 'clases' ya es sujeto de 'comienzan',
        por lo que no se debe inferir ELLOS.
        """
        verb_id = verb_tok.get("id")
        verb_uid = verb_tok.get("uid")
        sent_id = verb_tok.get("sent_id")

        subject_deps = {
            "nsubj",
            "csubj",
            "nsubj:pass",
        }

        for tok in original_tokens:
            if tok.get("sent_id") != sent_id:
                continue

            deprel = str(
                tok.get("deprel", "")
            ).lower()

            if deprel not in subject_deps:
                continue

            child_head = tok.get("head")
            child_head_uid = tok.get("head_uid")

            if child_head == verb_id:
                return True

            if child_head == verb_uid:
                return True

            if child_head_uid == verb_uid:
                return True

            if str(child_head) == str(verb_id):
                return True

            if str(child_head) == str(verb_uid):
                return True

        return False



    def infer_pronoun_from_verb_features(
        self,
        tok: Dict[str, Any],
    ) -> Optional[str]:
        """
        Infiere un pronombre a partir de la persona y número
        de un verbo finito.

        Mapeo usado:
            1 Sing -> yo
            1 Plur -> nosotros
            2 Sing -> tu
            2 Plur -> ustedes
            3 Sing -> el
            3 Plur -> ellos

        Nota:
        En español, la tercera persona puede ser ambigua:
            tiene  -> él / ella / usted
            tienen -> ellos / ellas / ustedes

        Para este sistema se usa:
            tiene  -> EL
            tienen -> ELLOS

        Si el sujeto aparece explícito, no se infiere.
        """
        feats = tok.get("feats_dict", {})

        person = str(feats.get("Person", "")).strip()
        number = str(feats.get("Number", "")).strip()

        person_number_to_pronoun = {
            ("1", "Sing"): "yo",
            ("1", "Plur"): "nosotros",

            ("2", "Sing"): "tu",
            ("2", "Plur"): "ustedes",

            ("3", "Sing"): "el",
            ("3", "Plur"): "ellos",
        }

        return person_number_to_pronoun.get(
            (person, number)
        )


    def detect_subordinate_subject_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set,
    ) -> List[Dict[str, Any]]:

        out: List[Dict[str, Any]] = []

        # Verificar si la oración contiene un contexto tipo:
        # "me enteré de que ..."
        has_me_before_enterar = False

        for tok in original_tokens:
            if tok.get("lemma") not in {"enterar", "enterarse"}:
                continue

            tok_uid = tok.get("uid", 0)

            has_me = any(
                t.get("uid", 0) < tok_uid
                and t.get("norm") == "me"
                for t in original_tokens
            )

            if has_me:
                has_me_before_enterar = True
                break

        if not has_me_before_enterar:
            return out

        for tok in original_tokens:
            if tok["uid"] in blocked_uids:
                continue

            if tok["upos"] not in {"VERB", "AUX"}:
                continue

            feats = tok.get("feats_dict", {})

            if feats.get("VerbForm") != "Fin":
                continue

            # Nos interesa el verbo subordinado: tienes/tienen/tiene.
            if tok.get("lemma") != "tener":
                continue

            # Debe aparecer cerca de un "que" anterior:
            # de que tienes / de que tienen.
            has_que_before = any(
                t.get("sent_id") == tok.get("sent_id")
                and 0 < tok.get("uid", 0) - t.get("uid", 0) <= 3
                and t.get("norm") == "que"
                for t in original_tokens
            )

            if not has_que_before:
                continue

            has_explicit_subject = any(
                child.get("sent_id") == tok.get("sent_id")
                and child.get("head") == tok.get("id")
                and child.get("deprel") in {
                    "nsubj",
                    "csubj",
                    "nsubj:pass",
                }
                for child in original_tokens
            )

            if has_explicit_subject:
                continue

            if self.verb_has_explicit_subject(tok, original_tokens):
                continue

            pron = self.infer_pronoun_from_verb_features(tok)

            if pron is None:
                continue

            cand = self.synthetic_candidate(
                pron,
                source="subject_inferred_subordinate",
                upos="PRON",
            )

            cand["anchor_uid"] = tok["uid"]

            out.append(cand)

        return out

    def detect_subject_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set,
        has_existing_subject: bool = False
    ) -> List[Dict[str, Any]]:
        out = []

        if has_existing_subject:
            return out

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            subject_like_pron = (
                tok["norm"] in EXPLICIT_PRONOUN_FORMS
                and tok["deprel"] in {"nsubj", "csubj"}
            )

            if tok["deprel"] in {"nsubj", "csubj"} and tok["upos"] in {"NOUN", "PROPN", "PRON"}:
                out.append(self.token_candidate(tok, source="subject"))
            elif subject_like_pron:
                out.append(self.token_candidate(tok, source="subject"))

        if not out:
            out.extend(self.infer_subject_from_verb(original_tokens, blocked_uids))

        return out

    def detect_object_candidates(self, kept_tokens: List[Dict[str, Any]], blocked_uids: set) -> List[Dict[str, Any]]:
        out = []
        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue
            if tok["deprel"] in {"obj", "iobj", "obl"} and tok["upos"] in {"NOUN", "PROPN", "PRON"}:
                out.append(self.token_candidate(tok, source="object"))
        return out

    def detect_enclitic_object_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> List[Dict[str, Any]]:

        out: List[Dict[str, Any]] = []

        for tok in original_tokens:
            if tok["uid"] in blocked_uids:
                continue

            if tok.get("upos") not in {"VERB", "AUX"}:
                continue


            feats_dict = tok.get("feats_dict", {}) or {}
            verb_form = feats_dict.get("VerbForm")

            if verb_form != "Inf":
                continue

            surface = normalize_token(
                tok.get("text", "") or tok.get("norm", "")
            )

            base, cliticos = separar_verbo_y_cliticos(surface)
            if not base or not cliticos:
                continue

            # Verificación adicional: el infinitivo recuperado debe
            # coincidir con el lema operativo de spaCy cuando este existe.
            lemma = normalize_token(tok.get("lemma", ""))
            if lemma and lemma.endswith(("ar", "er", "ir")) and base != lemma:
                continue

            for index, clitico in enumerate(cliticos):
                pronombre = ENCLITIC_TO_PRONOUN.get(clitico)

                # "se" no se expande automáticamente.
                if not pronombre:
                    continue

                cand = self.synthetic_candidate(
                    pronombre,
                    source="enclitic_object",
                    upos="PRON",
                )
                cand["anchor_uid"] = tok["uid"]
                cand["anchor_offset"] = 0.1 + (index * 0.01)
                cand["clitic"] = clitico
                cand["verb_base"] = base
                cand["original_verb"] = surface
                out.append(cand)

        return out

    def detect_conjunction_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> List[Dict[str, Any]]:
        out = []

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            norm = tok["norm"]

            if norm in PARTIAL_CONJUNCTIONS or norm in self.connector_words:
                cand = self.token_candidate(
                    tok,
                    source="conjunction",
                    token_override=norm
                )
                cand["connector_role"] = self.classify_connector_role_from_text(norm)
                out.append(cand)

        return out


    def detect_quantified_time_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> Tuple[List[Dict[str, Any]], set]:


        out: List[Dict[str, Any]] = []
        used_uids = set()

        i = 0

        while i < len(original_tokens):
            prep = original_tokens[i]

            if prep["uid"] in blocked_uids:
                i += 1
                continue

            # En esta primera versión se reconoce:
            # en + número + unidad temporal
            if prep["norm"] != "en":
                i += 1
                continue

            j = i + 1

            # Permitir determinantes intermedios:
            # "en unos tres minutos"
            while (
                j < len(original_tokens)
                and original_tokens[j]["upos"] == "DET"
            ):
                if original_tokens[j]["uid"] in blocked_uids:
                    break
                j += 1

            if j >= len(original_tokens):
                i += 1
                continue

            number_tok = original_tokens[j]

            if (
                number_tok["uid"] in blocked_uids
                or number_tok["upos"] != "NUM"
            ):
                i += 1
                continue

            k = j + 1

            if k >= len(original_tokens):
                i += 1
                continue

            unit_tok = original_tokens[k]

            if unit_tok["uid"] in blocked_uids:
                i += 1
                continue

            unit_lemma = normalize_token(
                unit_tok.get("lemma", "")
                or unit_tok.get("norm", "")
            )

            unit_surface = normalize_token(
                unit_tok.get("norm", "")
                or unit_tok.get("text", "")
            )

            if (
                unit_lemma not in TIME_UNIT_WORDS
                and unit_surface not in TIME_UNIT_WORDS
            ):
                i += 1
                continue

            number_candidate = self.token_candidate(
                number_tok,
                source="time_quantity"
            )
            number_candidate["time_role"] = "scene_time"

            unit_candidate = self.token_candidate(
                unit_tok,
                source="time_unit"
            )
            unit_candidate["time_role"] = "scene_time"

            out.extend([
                number_candidate,
                unit_candidate,
            ])

            # Consumir:
            # en + posibles determinantes + número + unidad
            used_uids.add(prep["uid"])

            for pos in range(i + 1, k + 1):
                used_uids.add(original_tokens[pos]["uid"])

            i = k + 1

        return out, used_uids



    def detect_preposition_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> List[Dict[str, Any]]:
        out = []
        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            if tok["norm"] in self.preposition_words:
                out.append(self.token_candidate(tok, source="preposition", token_override=tok["norm"]))

        return out

    def detect_adverb_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set,
        mode: str
    ) -> List[Dict[str, Any]]:
        out = []
        neg_words = {"no", "nunca", "jamás", "jamas"}

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            norm = tok["norm"]
            prev_tok = self.previous_non_punct_token(kept_tokens, tok["uid"])
            follows_preposition = prev_tok and prev_tok["norm"] in self.preposition_words

            if norm in neg_words:
                continue
            if norm in self.connector_words:
                continue
            if mode == "interrogative" and norm in QUESTION_CANONICAL_MAP:
                continue

            if tok["upos"] == "ADV":
                if norm in TIME_WORDS and not follows_preposition:
                    continue

                out.append(self.token_candidate(tok, source="adverb"))
                continue

            if norm in TIME_WORDS and follows_preposition:
                out.append(self.token_candidate(tok, source="adverb"))

        return out


    def detect_existential_haber_time_candidates(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set,
        has_explicit_time: bool
    ) -> Tuple[List[Dict[str, Any]], set]:

        return [], set()

    def detect_sentence_tense_candidate(
        self,
        original_tokens: List[Dict[str, Any]],
        blocked_uids: set,
        has_explicit_time: bool
    ) -> List[Dict[str, Any]]:

        return []

    def detect_main_verb_candidates(
        self,
        kept_tokens: List[Dict[str, Any]],
        blocked_uids: set
    ) -> List[Dict[str, Any]]:
        out = []

        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            if (
                tok["upos"] == "VERB"
                and tok["deprel"] in {
                    "root", "conj", "xcomp", "ccomp", "advcl"
                }
            ):
                out.append(self.token_candidate(tok, source="verb"))

        if not out:
            for tok in kept_tokens:
                if tok["uid"] in blocked_uids:
                    continue
                if tok["upos"] == "VERB":
                    out.append(self.token_candidate(tok, source="verb"))

        return out

    def detect_residual_candidates(self, kept_tokens: List[Dict[str, Any]], blocked_uids: set) -> List[Dict[str, Any]]:
        out = []
        for tok in kept_tokens:
            if tok["uid"] in blocked_uids:
                continue

            # Evitar que los posesivos prenominales salgan como glosa residual
            if tok["norm"] in POSSESSIVE_TO_OWNER_PRONOUN:
                continue

            out.append(self.token_candidate(tok, source="residual"))
        return out


    def extract_structures(self, text: str) -> Dict[str, Any]:
        original_tokens = self.analyze(text)
        mode = self.detect_sentence_mode(text, original_tokens)

        kept_tokens = [t for t in original_tokens if self.keep_token(t, mode)]

        interrogative_candidates = self.detect_interrogative_candidates(
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            mode=mode
        )

        consumed_uids = set()
        for c in interrogative_candidates:
            if c.get("span_uids"):
                consumed_uids.update(c["span_uids"])
            else:
                uid = c.get("uid")
                if isinstance(uid, int) and uid > 0:
                    consumed_uids.add(uid)


        quantified_time_candidates, quantified_time_uids = (
            self.detect_quantified_time_candidates(
                original_tokens,
                blocked_uids=consumed_uids
            )
        )

        time_candidates = list(quantified_time_candidates)
        consumed_uids.update(quantified_time_uids)

        # ---------------------------------------------------------
        # Expresiones simples:
        # ayer, hoy, mañana, noche...
        # ---------------------------------------------------------
        simple_time_candidates = self.detect_time_candidates(
            kept_tokens,
            blocked_uids=consumed_uids
        )

        time_candidates.extend(simple_time_candidates)

        consumed_uids.update(
            c["uid"]
            for c in simple_time_candidates
            if isinstance(c.get("uid"), int)
            and c["uid"] > 0
        )

        has_explicit_time = len(time_candidates) > 0


        consumed_uids.update(
            c["uid"]
            for c in time_candidates
            if isinstance(c.get("uid"), int)
            and c["uid"] > 0
        )

        negation_candidates = self.detect_negation_candidates(
            kept_tokens,
            blocked_uids=consumed_uids
        )

        consumed_uids.update(
            c["uid"]
            for c in negation_candidates
        )

        existential_haber_time, haber_uids = (
            self.detect_existential_haber_time_candidates(
                original_tokens,
                blocked_uids=consumed_uids,
                has_explicit_time=has_explicit_time,
            )
        )

        time_candidates.extend(existential_haber_time)
        consumed_uids.update(haber_uids)



        phrase_buckets, phrase_candidates, phrase_uids = self.detect_phrase_candidates(
            original_tokens,
            blocked_uids=consumed_uids
        )
        consumed_uids.update(phrase_uids)

        possession_candidates, possession_uids = self.detect_possession_candidates(
            original_tokens,
            blocked_uids=consumed_uids
        )

        possessive_det_candidates, possessive_det_uids = self.detect_possessive_determiner_candidates(
            original_tokens,
            blocked_uids=consumed_uids
        )
        possession_candidates.extend(possessive_det_candidates)

        consumed_uids.update(possession_uids)
        consumed_uids.update(possessive_det_uids)

        place_candidates = list(phrase_buckets["place"])
        conjunction_candidates = list(phrase_buckets["conjunctions"])
        preposition_candidates = list(phrase_buckets["prepositions"])
        subject_candidates = list(phrase_buckets["subject"])
        object_candidates = list(phrase_buckets["object"])
        adverb_candidates = list(phrase_buckets["adverbs"])
        verb_candidates = list(phrase_buckets["verbs"])
        residual_candidates = list(phrase_buckets["residual"])

        place_more, place_uids = self.detect_place_candidates(original_tokens, blocked_uids=consumed_uids)
        place_candidates.extend(place_more)
        consumed_uids.update(place_uids)

        subject_more = self.detect_subject_candidates(
            kept_tokens=kept_tokens,
            original_tokens=original_tokens,
            blocked_uids=consumed_uids,
            has_existing_subject=(len(subject_candidates) > 0)
        )
        subject_candidates.extend(subject_more)

        for cand in subject_more:
            consumed_uids.update(
                cand.get("linked_pronoun_uids", set())
            )

        consumed_uids.update(
            c["uid"] for c in subject_more
            if isinstance(c["uid"], int) and c["uid"] > 0
        )

        subordinate_subject_more = (
            self.detect_subordinate_subject_candidates(
                original_tokens,
                blocked_uids=consumed_uids,
            )
        )

        subject_candidates.extend(subordinate_subject_more)

        object_more = self.detect_object_candidates(kept_tokens, blocked_uids=consumed_uids)
        object_candidates.extend(object_more)
        consumed_uids.update(c["uid"] for c in object_more)

        enclitic_object_more = self.detect_enclitic_object_candidates(
            original_tokens=original_tokens,
            blocked_uids=consumed_uids,
        )
        object_candidates.extend(enclitic_object_more)

        conjunction_more = self.detect_conjunction_candidates(kept_tokens, blocked_uids=consumed_uids)
        conjunction_candidates.extend(conjunction_more)
        consumed_uids.update(c["uid"] for c in conjunction_more)

        preposition_more = self.detect_preposition_candidates(kept_tokens, blocked_uids=consumed_uids)
        preposition_candidates.extend(preposition_more)
        consumed_uids.update(c["uid"] for c in preposition_more)

        adverb_more = self.detect_adverb_candidates(kept_tokens, blocked_uids=consumed_uids, mode=mode)
        adverb_candidates.extend(adverb_more)
        consumed_uids.update(c["uid"] for c in adverb_more)

        sentence_tense_candidates = (
            self.detect_sentence_tense_candidate(
                original_tokens,
                blocked_uids=consumed_uids,
                has_explicit_time=(
                    has_explicit_time
                    or bool(existential_haber_time)
                ),
            )
        )

        time_candidates.extend(sentence_tense_candidates)
        time_candidates = self.normalize_time_candidates(
            time_candidates
        )

        verb_more = self.detect_main_verb_candidates(kept_tokens, blocked_uids=consumed_uids)
        verb_candidates.extend(verb_more)
        consumed_uids.update(c["uid"] for c in verb_more)

        residual_more = self.detect_residual_candidates(kept_tokens, blocked_uids=consumed_uids)
        residual_candidates.extend(residual_more)

        return {
            "mode": mode,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "phrases": phrase_candidates,
            "interrogative": interrogative_candidates,
            "time": time_candidates,
            "negation": negation_candidates,
            "possession": possession_candidates,
            "place": place_candidates,
            "subject": subject_candidates,
            "object": object_candidates,
            "conjunctions": conjunction_candidates,
            "prepositions": preposition_candidates,
            "adverbs": adverb_candidates,
            "verbs": verb_candidates,
            "residual": residual_candidates,
        }

    # -----------------------------------------------------
    # 4) CONSTRUCCIÓN DE GLOSAS CANDIDATAS
    # -----------------------------------------------------
    def build_fixed_gloss_candidate(
        self,
        token: str,
        gloss: str,
        source: str,
        upos: str = "X"
    ) -> Dict[str, Any]:
        norm = normalize_token(token)
        return {
            "uid": self._next_uid(),
            "token": norm,
            "lookup_forms": [norm],
            "surface": norm,
            "lemma_form": norm,
            "upos": upos,
            "source": source,
            "text": gloss,
            "pre_resolved": {
                "input": norm,
                "resolved_key": norm,
                "gloss": gloss,
                "mode": "fixed_gloss",
                "score": 1.0,
                "source": source,
            }
        }

    def build_gloss_candidates_from_structure(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        mode = structure["mode"]

        front_time_meta = sorted(
            [c for c in structure["time"] if c["token"] in BUILTIN_META_GLOSSES],
            key=self.candidate_position_key
        )

        front_scene_time = sorted(
            [
                c for c in structure["time"]
                if c["token"] not in BUILTIN_META_GLOSSES
                and c.get("time_role", "scene_time") != "sequence_time"
            ],
            key=self.candidate_position_key
        )

        front_attitude = sorted(
            [
                c for c in structure["conjunctions"]
                if self.classify_connector_role_from_text(c.get("text") or c.get("token", "")) == "attitude"
            ],
            key=self.candidate_position_key
        )

        if mode == "interrogative" and structure["interrogative"]:
            # 1. Marcador técnico de pregunta
            ordered.append(
                self.build_fixed_gloss_candidate(
                    token="pregunta",
                    gloss="PREGUNTA",
                    source="question_marker",
                    upos="PART"
                )
            )

            # 2. Marcadores temporales inferidos:
            # PASADO, FUTURO
            ordered.extend(front_time_meta)

            # 3. Marcadores temporales explícitos:
            # AYER, HOY, MAÑANA, NOCHE, etc.
            ordered.extend(front_scene_time)

            # 4. Palabra interrogativa:
            # DONDE, QUE, QUIEN, CUANTO, COMO...
            ordered.extend(structure["interrogative"])

            # 5. Conectores de actitud
            ordered.extend(front_attitude)


            already = {
                self.candidate_identity(c)
                for c in ordered
            }

            def add_bucket(bucket_name: str) -> None:
                candidates = sorted(
                    structure[bucket_name],
                    key=self.candidate_position_key,
                )

                for cand in candidates:
                    identity = self.candidate_identity(cand)

                    if identity in already:
                        continue

                    ordered.append(cand)
                    already.add(identity)


            # Orden estructural para interrogativas:
            # INTERROGATIVO + SUJETO + VERBO + OBJETO
            add_bucket("subject")
            add_bucket("verbs")
            add_bucket("object")
            add_bucket("possession")
            add_bucket("place")
            add_bucket("prepositions")
            add_bucket("adverbs")
            add_bucket("conjunctions")
            add_bucket("time")
            add_bucket("residual")
            add_bucket("negation")


        else:
            ordered.extend(front_time_meta)
            ordered.extend(front_scene_time)
            ordered.extend(front_attitude)

            already = {self.candidate_identity(c) for c in ordered}

            remainder = []
            for bucket_name in [
                "possession", "subject", "object",
                "verbs", "prepositions", "place", "adverbs",
                "conjunctions", "time", "residual", "negation"
            ]:
                remainder.extend(structure[bucket_name])

            remainder = [c for c in remainder if self.candidate_identity(c) not in already]
            remainder.sort(key=self.candidate_position_key)
            ordered.extend(remainder)

        seen = set()
        final = []
        for cand in ordered:
            key = self.candidate_identity(cand)
            if key not in seen and cand.get("token", ""):
                seen.add(key)
                final.append(cand)

        return final

    # -----------------------------------------------------
    # 5) RESOLUCIÓN FINAL
    # -----------------------------------------------------

    def resolve_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prioridad de resolución:

        1. Lema operativo en glosses.txt.
        2. Base verbal derivada directamente de la superficie.
        3. Forma superficial en glosses.txt.
        4. Otras formas derivadas de lookup_forms.
        5. Sinónimo manual.
        6. DLT o [UNK].
        """
        if "pre_resolved" in candidate:
            return candidate["pre_resolved"]

        lookup_forms = candidate.get("lookup_forms", [])
        upos = candidate.get("upos", "")
        source = candidate.get("source", "")

        surface = normalize_token(
            candidate.get("surface", "")
            or candidate.get("token", "")
            or (lookup_forms[0] if lookup_forms else "")
        )
        lemma = normalize_token(candidate.get("lemma_form", ""))

        # Reglas internas.
        if source == "interrogative":
            interrogative_token = lemma or surface
            return {
                "input": surface,
                "resolved_key": interrogative_token,
                "gloss": interrogative_token.upper(),
                "mode": "interrogative_builtin",
                "score": 1.0,
                "source": source,
            }

        if source == "conjunction" and surface in PARTIAL_CONJUNCTIONS:
            return {
                "input": surface,
                "resolved_key": surface,
                "gloss": surface.upper(),
                "mode": "conjunction_builtin",
                "score": 1.0,
                "source": source,
            }

        builtin_key = lemma or surface
        if builtin_key in BUILTIN_META_GLOSSES:
            return {
                "input": surface,
                "resolved_key": builtin_key,
                "gloss": BUILTIN_META_GLOSSES[builtin_key],
                "mode": "time_builtin",
                "score": 1.0,
                "source": source,
            }

        if upos == "PRON":
            pronoun_key = PERSONAL_PRONOUN_TO_GLOSS.get(surface)

            if pronoun_key and pronoun_key in self.gloss_lexicon:
                return {
                    "input": surface,
                    "resolved_key": pronoun_key,
                    "gloss": self.gloss_lexicon[pronoun_key],
                    "mode": "personal_pronoun_surface",
                    "score": 1.0,
                    "source": source,
                }

        if lemma and lemma in self.gloss_lexicon:
            return {
                "input": surface,
                "resolved_key": lemma,
                "gloss": self.gloss_lexicon[lemma],
                "mode": "lemma",
                "score": 1.0,
                "source": source,
            }

        # 2) Base directa de un verbo con clítico.
        direct_base = None
        if upos in {"VERB", "AUX"}:
            direct_base = verbo_con_cliticos_a_base(surface)

        if direct_base and direct_base in self.gloss_lexicon:
            return {
                "input": surface,
                "resolved_key": direct_base,
                "gloss": self.gloss_lexicon[direct_base],
                "mode": "enclitic_base",
                "score": 1.0,
                "source": source,
            }

        # 3) Forma superficial.
        if surface and surface in self.gloss_lexicon:
            return {
                "input": surface,
                "resolved_key": surface,
                "gloss": self.gloss_lexicon[surface],
                "mode": "surface",
                "score": 1.0,
                "source": source,
            }

        # 4) Otras formas derivadas.
        derived_forms: List[str] = []
        for form in [direct_base, *lookup_forms]:
            form_norm = normalize_token(form or "")
            if (
                form_norm
                and form_norm not in {lemma, surface}
                and form_norm not in derived_forms
            ):
                derived_forms.append(form_norm)

        for form in derived_forms:
            if form in self.gloss_lexicon:
                return {
                    "input": surface,
                    "resolved_key": form,
                    "gloss": self.gloss_lexicon[form],
                    "mode": "derived_form",
                    "score": 1.0,
                    "source": source,
                }

        # 5) Sinónimo manual. Primero se prueba también la base enclítica.
        if direct_base:
            entry = self.manual_synonym_map.get(direct_base)
            if entry:
                return {
                    "input": surface,
                    "resolved_key": direct_base,
                    "gloss": entry["gloss"],
                    "mode": "manual_synonym_enclitic_base",
                    "score": 1.0,
                    "source": source,
                    "raw_left": entry.get("raw_left"),
                    "raw_right": entry.get("raw_right"),
                }

        manual_syns = self.manual_synonym_candidates(candidate)
        if manual_syns:
            chosen = manual_syns[0]
            return {
                "input": surface,
                "resolved_key": chosen["matched_form"],
                "gloss": chosen["gloss"],
                "mode": "manual_synonym",
                "score": 1.0,
                "source": source,
                "raw_left": chosen.get("raw_left"),
                "raw_right": chosen.get("raw_right"),
                "manual_synonym_candidates": manual_syns,
            }

        # 6) DLT. Si se reconoció la base verbal, se deletrea la base y no
        # la forma completa con el pronombre enclítico.
        fallback_token = direct_base or surface

        if self.unknown_mode == "spell":
            return {
                "input": surface,
                "resolved_key": fallback_token,
                "gloss": spell_gloss(fallback_token),
                "mode": "spell_base" if direct_base else "spell",
                "score": 0.0,
                "source": source,
            }

        return {
            "input": surface,
            "resolved_key": fallback_token,
            "gloss": "[UNK]",
            "mode": "unk",
            "score": 0.0,
            "source": source,
        }

    def translate(self, text: str) -> Dict[str, Any]:
        correction = self.correct_text(text)
        corrected_text = correction["corrected_text"]

        structure = self.extract_structures(corrected_text)
        gloss_candidates = self.build_gloss_candidates_from_structure(structure)

        raw_matches = []
        for cand in gloss_candidates:
            resolved = self.resolve_candidate(cand)
            raw_matches.append(resolved)

        matches = self.collapse_consecutive_resolved_matches(raw_matches)
        final_glosses = [m["gloss"] for m in matches]

        if structure["mode"] == "interrogative" and structure["interrogative"]:
            if not final_glosses or final_glosses[0] != "PREGUNTA":
                final_glosses = ["PREGUNTA"] + [g for g in final_glosses if g != "PREGUNTA"]

        return {
            "input_text": text,
            "corrected_text": corrected_text,
            "num_lt_matches_found": correction["num_matches_found"],
            "num_lt_matches_applied": correction["num_matches_applied"],
            "applied_lt_rules": correction["applied_rules"],
            "mode": structure["mode"],
            "gloss_candidates": [c["token"] for c in gloss_candidates],
            "glosses": final_glosses,
            "matches": matches,
            "structure": structure,
        }

    def close(self) -> None:
        try:
            self.lt_tool.close()
        except Exception:
            pass