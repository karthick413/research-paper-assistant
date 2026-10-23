
import ollama


def _get_field(obj, key, default=None):
    """Read a field from either a dictionary or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def list_model_names() -> set[str]:
    """Return the model names currently available in Ollama."""
    response = ollama.list()
    models = _get_field(response, "models", []) or []

    names = set()

    for model in models:
        name = (
            _get_field(model, "model")
            or _get_field(model, "name")
        )

        if name:
            names.add(str(name).strip())

    return names


def model_is_available(requested_model: str, available_models: set[str]) -> bool:
    """Check model availability, accepting an omitted :latest tag."""
    requested = requested_model.strip()

    if requested in available_models:
        return True

    if ":" not in requested:
        return f"{requested}:latest" in available_models

    if requested.endswith(":latest"):
        return requested.removesuffix(":latest") in available_models

    return False


def check_required_models(
    llm_model: str,
    embed_model: str,
) -> dict:
    """Check whether both the chat and embedding models are installed."""
    try:
        available = list_model_names()

        return {
            "ok": (
                model_is_available(llm_model, available)
                and model_is_available(embed_model, available)
            ),
            "available": sorted(available),
            "missing": [
                model
                for model in (llm_model, embed_model)
                if not model_is_available(model, available)
            ],
            "error": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "available": [],
            "missing": [llm_model, embed_model],
            "error": str(exc),
        }


def embed_texts(
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Generate embeddings for a list of text strings."""
    if not texts:
        return []

    response = ollama.embed(
        model=model,
        input=texts,
    )

    embeddings = _get_field(response, "embeddings")

    if embeddings is None:
        raise RuntimeError(
            "Ollama returned no embeddings. Check the model and Ollama service."
        )

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, received {len(embeddings)}."
        )

    return [
        [float(value) for value in embedding]
        for embedding in embeddings
    ]


def chat_completion(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
) -> str:
    """Generate a chat response using Ollama."""
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )

    message = _get_field(response, "message")

    if message is None:
        raise RuntimeError("Ollama returned no chat message.")

    content = _get_field(message, "content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty chat response.")

    return content.strip()
