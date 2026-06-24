"""Parser pour le format GGUF (GPT-Generated Unified Format).

Structure du header GGUF v3 :
  Offset  Taille  Contenu
  0       4       magic: 'GGUF' (0x46554747)
  4       4       version: uint32
  8       8       tensor_count: uint64
  16      8       metadata_kv_count: uint64
  24      ?       metadata_kv[] : array de paires clé-valeur
  ...     ?       tensor_infos[] (pas parsé ici)

Types de valeurs GGUF :
  0: uint8     1: int8      2: uint16    3: int16
  4: uint32    5: int32     6: float32   7: bool
  8: string    9: array     10: uint64   11: int64
  12: float64

Ce parser ne nécessite aucune base de modèles connus.
Toutes les métadonnées sont extraites structurellement du fichier.
"""

import os
import struct
from typing import Any, BinaryIO, Optional

from app.models import ModelMeta

# ─── Constantes GGUF ───────────────────────────────────

GGUF_MAGIC = b"GGUF"
GGUF_VERSION_MIN = 2
GGUF_VERSION_MAX = 3

# Types de valeurs
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Mapping type_id → struct format
TYPE_FORMATS = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

# Mapping nom quant → file_type ID (GGUF standard)
GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",  # Legacy
    5: "Q4_2",           # Legacy
    6: "Q4_3",           # Legacy
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "IQ1_M",
}


# ─── Lecture bas niveau ────────────────────────────────


def _read_struct(f: BinaryIO, fmt: str):
    """Lit et unpack une structure binaire."""
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) < size:
        raise ValueError(f"Fin de fichier inattendue en lisant {fmt} (lu {len(data)}/{size} octets)")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    """Lit une chaîne GGUF : length (uint64) + data (UTF-8)."""
    length = _read_struct(f, "<Q")
    data = f.read(length)
    if len(data) < length:
        raise ValueError(f"Fin de fichier inattendue en lisant une chaîne de {length} octets")
    return data.decode("utf-8")


def _read_value(f: BinaryIO, type_id: int) -> Any:
    """Lit une valeur GGUF en fonction de son type."""
    if type_id == GGUF_TYPE_STRING:
        return _read_string(f)
    elif type_id == GGUF_TYPE_ARRAY:
        array_type = _read_struct(f, "<I")
        array_length = _read_struct(f, "<Q")
        return [_read_value(f, array_type) for _ in range(array_length)]
    elif type_id == GGUF_TYPE_BOOL:
        return bool(_read_struct(f, "B"))
    elif type_id in TYPE_FORMATS:
        return _read_struct(f, TYPE_FORMATS[type_id])
    else:
        raise ValueError(f"Type GGUF inconnu : {type_id}")


# ─── Parser principal ─────────────────────────────────


def parse_gguf_header(filepath: str) -> dict[str, Any]:
    """Parse le header d'un fichier GGUF et retourne les métadonnées.

    Lit uniquement le header (pas les tenseurs), donc rapide même
    sur des fichiers de plusieurs Go.

    Args:
        filepath: Chemin vers le fichier .gguf

    Returns:
        Dictionnaire des métadonnées (clés GGUF standards)

    Raises:
        ValueError: Si le format est invalide ou non supporté
        FileNotFoundError: Si le fichier n'existe pas
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    file_size = os.path.getsize(filepath)
    metadata: dict[str, Any] = {
        "file_size": file_size,
        "file_size_gb": round(file_size / 1_000_000_000, 2),
    }

    with open(filepath, "rb") as f:
        # Magic number
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Magic GGUF invalide : {magic!r}")

        # Version
        version = _read_struct(f, "<I")
        if version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX:
            raise ValueError(
                f"Version GGUF {version} non supportée "
                f"(supportée : {GGUF_VERSION_MIN}-{GGUF_VERSION_MAX})"
            )
        metadata["version"] = version

        # Tensor count (pas utilisé mais on lit pour avancer)
        tensor_count = _read_struct(f, "<Q")
        metadata["tensor_count"] = tensor_count

        # Metadata count
        metadata_count = _read_struct(f, "<Q")

        # Lecture des paires clé-valeur
        for _ in range(metadata_count):
            key = _read_string(f)
            type_id = _read_struct(f, "<I")
            value = _read_value(f, type_id)
            metadata[key] = value

    return metadata


# ─── Conversion en ModelMeta ───────────────────────────


def metadata_to_model_meta(metadata: dict, model_id: Optional[str] = None) -> ModelMeta:
    """Convertit le dictionnaire brut de métadonnées en ModelMeta structuré.

    Extrait et normalise les champs depuis les clés GGUF standards.
    Calcule les valeurs dérivées (MoE, paramètres actifs, etc.).
    """
    architecture = str(metadata.get("general.architecture", ""))
    name = str(metadata.get("general.name", ""))
    file_type = int(metadata.get("general.file_type", 0))
    param_count = int(metadata.get("general.parameter_count", 0))
    block_count = int(metadata.get(architecture + ".block_count", 0))
    context_length = int(metadata.get(architecture + ".context_length", 0))
    embedding_length = int(metadata.get(architecture + ".embedding_length", 0))
    head_count = int(metadata.get(architecture + ".attention.head_count", 0))
    expert_count = int(metadata.get(architecture + ".expert_count", 0))
    expert_used_count = int(metadata.get(architecture + ".expert_used_count", 0))

    # Conversion du file_type en nom de quant lisible
    quant_name = GGUF_FILE_TYPES.get(file_type, f"Unknown({file_type})")

    # Détection MoE
    is_moe = expert_count > 1

    # Si param_count est 0 (parfois absent des métadonnées), l'estimer
    # depuis la taille distante du fichier et le type de quant
    if param_count == 0:
        remote_size = metadata.get("_remote_size", 0)
        if remote_size > 0:
            bits_per = _file_type_to_bits(file_type)
            if bits_per > 0:
                param_count = int(remote_size * 8 / bits_per)
                metadata["general.parameter_count"] = param_count

    # Calcul des paramètres actifs (pour MoE)
    active_params_b: float = 0.0
    if param_count > 0 and is_moe and expert_count > 0 and expert_used_count > 0:
        # Approximation : ~10% des paramètres sont dans l'attention
        # Les ~90% sont dans les experts
        # actifs = attention + (experts_used / experts_total) * experts
        expert_params = param_count * 0.9
        active_params = param_count * 0.1 + (expert_used_count / expert_count) * expert_params
        active_params_b = round(active_params / 1_000_000_000, 2)
    elif param_count > 0:
        active_params_b = round(param_count / 1_000_000_000, 2)

    params_b = round(param_count / 1_000_000_000, 2) if param_count > 0 else 0.0

    return ModelMeta(
        id=model_id or name or "",
        path=metadata.get("_filepath", ""),
        file_size_gb=metadata.get("file_size_gb", 0.0),
        architecture=architecture,
        name=name,
        quant=quant_name,
        param_count=param_count,
        params_b=params_b,
        is_moe=is_moe,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        active_params_b=active_params_b,
        block_count=block_count,
        context_length=context_length,
        embedding_length=embedding_length,
        head_count=head_count,
    )


def _file_type_to_bits(file_type: int) -> float:
    """Estime le nombre de bits par paramètre pour un GGUF file_type."""
    bits_map = {
        0: 32.0, 1: 16.0, 2: 4.0, 3: 4.5,
        4: 4.5, 5: 4.5, 6: 4.5, 7: 8.0,
        8: 5.0, 9: 5.5, 10: 2.5, 11: 3.5,
        12: 4.5, 13: 5.5, 14: 6.5, 15: 8.5,
        16: 2.25, 17: 2.5, 18: 3.25, 19: 1.5,
        20: 4.5, 21: 3.5, 22: 2.5, 23: 4.5, 24: 1.75,
    }
    return bits_map.get(file_type, 4.5)


# ─── Fonction utilitaire pour parser depuis des bytes ──


def parse_gguf_header_from_bytes(data: bytes) -> dict[str, Any]:
    """Parse le header GGUF depuis un buffer bytes.

    Utilisé pour sonder un modèle distant sans le télécharger entièrement.
    Le buffer doit contenir au moins les premiers 50 Ko du fichier.
    """
    metadata: dict[str, Any] = {
        "file_size": len(data),
        "file_size_gb": round(len(data) / 1_000_000_000, 2),
    }

    offset = 0

    def read_bytes(n: int) -> bytes:
        nonlocal offset
        if offset + n > len(data):
            raise ValueError(
                f"Buffer trop court : besoin de {offset + n} octets, "
                f"disponible {len(data)}"
            )
        result = data[offset : offset + n]
        offset += n
        return result

    def read_fmt(fmt: str):
        size = struct.calcsize(fmt)
        raw = read_bytes(size)
        return struct.unpack(fmt, raw)[0]

    def read_string() -> str:
        length = read_fmt("<Q")
        raw = read_bytes(length)
        return raw.decode("utf-8")

    def read_value(type_id: int):
        """Lit une valeur GGUF récursivement."""
        if type_id == GGUF_TYPE_STRING:
            return read_string()
        elif type_id == GGUF_TYPE_ARRAY:
            array_type = read_fmt("<I")
            array_length = read_fmt("<Q")
            return [read_value(array_type) for _ in range(array_length)]
        elif type_id == GGUF_TYPE_BOOL:
            return bool(read_fmt("B"))
        elif type_id in TYPE_FORMATS:
            return read_fmt(TYPE_FORMATS[type_id])
        else:
            raise ValueError(f"Type GGUF inconnu : {type_id}")

    # Magic
    magic = read_bytes(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"Magic GGUF invalide : {magic!r}")

    # Version
    version = read_fmt("<I")
    if version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX:
        raise ValueError(f"Version GGUF {version} non supportée")
    metadata["version"] = version

    # Tensor count
    metadata["tensor_count"] = read_fmt("<Q")

    # Metadata count
    metadata_count = read_fmt("<Q")

    # Lecture des métadonnées
    for _ in range(metadata_count):
        key = read_string()
        type_id = read_fmt("<I")
        value = read_value(type_id)
        metadata[key] = value

    return metadata
