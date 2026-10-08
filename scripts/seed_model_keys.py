"""Seed an organization's model setups from AI provider keys in api/.env.

Creates (or updates) provider connections and up to two named model
configurations, then picks the organization default -- the same writes the
Providers and Models pages make. Safe to re-run after changing a key.

- "Azure OpenAI + Sarvam (.env)": pipeline mode. Azure OpenAI chat deployment
  for the LLM, Sarvam AI for STT and TTS. Built when
  AZURE_OPENAI_CHAT_DEPLOYMENT is set.
- "Azure OpenAI Realtime (.env)": speech-to-speech mode on an Azure realtime
  deployment. Built when AZURE_OPENAI_REALTIME_DEPLOYMENT is set. Its text LLM
  (variable extraction, voicemail detection) is the chat deployment, falling
  back to the realtime deployment when there is none.

Both get Azure embeddings for knowledge-base search when
AZURE_OPENAI_EMBEDDING_DEPLOYMENT is set.

Run from the repo root with the api environment loaded:

    python -m scripts.seed_model_keys --email you@example.com [--default realtime]

Reads from the environment:

    AZURE_OPENAI_API_KEY                required
    AZURE_OPENAI_ENDPOINT               required, e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_CHAT_DEPLOYMENT        chat model deployment name, e.g. gpt-4.1-mini
    AZURE_OPENAI_REALTIME_DEPLOYMENT    realtime deployment name, e.g. gpt-realtime-mini
    AZURE_OPENAI_REALTIME_VOICE         optional, default alloy
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT   optional, e.g. text-embedding-3-small
    SARVAM_API_KEY                      required for the pipeline setup
    SARVAM_STT_MODEL                    optional, default saarika:v2.5
    SARVAM_STT_LANGUAGE                 optional, default unknown (auto-detect)
    SARVAM_TTS_MODEL                    optional, default bulbul:v2
    SARVAM_TTS_VOICE                    optional, default anushka
    SARVAM_TTS_LANGUAGE                 optional, default hi-IN
"""

import argparse
import asyncio
import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from api.db import db_client
from api.services.configuration.model_connections import (
    connection_provider,
    resolve_inline_model_configuration,
    validate_provider_connection_credentials,
)

AZURE_CONNECTION_NAME = "Azure OpenAI (.env)"
AZURE_REALTIME_CONNECTION_NAME = "Azure OpenAI Realtime (.env)"
SARVAM_CONNECTION_NAME = "Sarvam AI (.env)"
PIPELINE_CONFIGURATION_NAME = "Azure OpenAI + Sarvam (.env)"
REALTIME_CONFIGURATION_NAME = "Azure OpenAI Realtime (.env)"


def _require(name: str) -> str:
    value = _optional(name, "")
    if not value:
        sys.exit(f"Missing {name} in api/.env")
    return value


def _optional(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _selection(connection, **settings) -> dict:
    return {"provider_connection_uuid": str(connection.uuid), "settings": settings}


async def _upsert_connection(
    organization_id, created_by, name, provider, credentials, settings
):
    await validate_provider_connection_credentials(
        provider,
        credentials,
        settings,
        organization_id=organization_id,
        created_by=created_by,
    )
    existing = next(
        (
            row
            for row in await db_client.list_provider_connections(organization_id)
            if row.name == name
        ),
        None,
    )
    if existing is None:
        row = await db_client.create_provider_connection(
            organization_id,
            name=name,
            provider=connection_provider(provider),
            credentials=credentials,
            connection_settings=settings,
        )
        print(f"Created connection: {name}")
    else:
        row = await db_client.update_provider_connection(
            organization_id,
            str(existing.uuid),
            changes={"credentials": credentials, "connection_settings": settings},
        )
        print(f"Updated connection: {name}")
    return row


async def _upsert_configuration(organization_id, name, configuration):
    await resolve_inline_model_configuration(organization_id, configuration)
    existing = next(
        (
            row
            for row in await db_client.list_named_model_configurations(organization_id)
            if row.name == name
        ),
        None,
    )
    if existing is None:
        row = await db_client.create_named_model_configuration(
            organization_id, name=name, configuration=configuration
        )
        print(f"Created model configuration: {name}")
    else:
        row = await db_client.update_named_model_configuration(
            organization_id,
            str(existing.uuid),
            changes={"configuration": configuration},
        )
        print(f"Updated model configuration: {name}")
    return row


async def main(email: str, default_mode: str | None) -> None:
    azure_key = _require("AZURE_OPENAI_API_KEY")
    azure_endpoint = _require("AZURE_OPENAI_ENDPOINT")
    chat_deployment = _optional("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    realtime_deployment = _optional("AZURE_OPENAI_REALTIME_DEPLOYMENT", "")
    embedding_deployment = _optional("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    if not chat_deployment and not realtime_deployment:
        sys.exit(
            "Set AZURE_OPENAI_CHAT_DEPLOYMENT and/or "
            "AZURE_OPENAI_REALTIME_DEPLOYMENT in api/.env"
        )

    user = await db_client.get_user_by_email(email)
    if user is None or user.selected_organization_id is None:
        sys.exit(
            f"No user with an organization found for {email}. Sign up in the app first."
        )
    organization_id = user.selected_organization_id

    azure = await _upsert_connection(
        organization_id,
        user.provider_id,
        AZURE_CONNECTION_NAME,
        "azure",
        {"api_key": azure_key},
        {"endpoint": azure_endpoint},
    )
    # The text LLM role: the chat deployment, or the realtime deployment as a
    # stand-in on a realtime-only resource.
    llm = _selection(azure, model=chat_deployment or realtime_deployment)
    embeddings = (
        _selection(azure, model=embedding_deployment) if embedding_deployment else None
    )

    built = {}
    if chat_deployment:
        sarvam = await _upsert_connection(
            organization_id,
            user.provider_id,
            SARVAM_CONNECTION_NAME,
            "sarvam",
            {"api_key": _require("SARVAM_API_KEY")},
            {},
        )
        built["pipeline"] = await _upsert_configuration(
            organization_id,
            PIPELINE_CONFIGURATION_NAME,
            {
                "version": 3,
                "mode": "pipeline",
                "llm": llm,
                "stt": _selection(
                    sarvam,
                    model=_optional("SARVAM_STT_MODEL", "saarika:v2.5"),
                    language=_optional("SARVAM_STT_LANGUAGE", "unknown"),
                ),
                "tts": _selection(
                    sarvam,
                    model=_optional("SARVAM_TTS_MODEL", "bulbul:v2"),
                    voice=_optional("SARVAM_TTS_VOICE", "anushka"),
                    language=_optional("SARVAM_TTS_LANGUAGE", "hi-IN"),
                ),
                "embeddings": embeddings,
            },
        )
    else:
        print(
            f"Skipped '{PIPELINE_CONFIGURATION_NAME}': "
            "set AZURE_OPENAI_CHAT_DEPLOYMENT to build it"
        )

    if realtime_deployment:
        azure_realtime = await _upsert_connection(
            organization_id,
            user.provider_id,
            AZURE_REALTIME_CONNECTION_NAME,
            "azure_realtime",
            {"api_key": azure_key},
            {"endpoint": azure_endpoint},
        )
        built["realtime"] = await _upsert_configuration(
            organization_id,
            REALTIME_CONFIGURATION_NAME,
            {
                "version": 3,
                "mode": "realtime",
                "llm": llm,
                "realtime": _selection(
                    azure_realtime,
                    model=realtime_deployment,
                    voice=_optional("AZURE_OPENAI_REALTIME_VOICE", "alloy"),
                ),
                "embeddings": embeddings,
            },
        )
        if not chat_deployment:
            print(
                "Note: without AZURE_OPENAI_CHAT_DEPLOYMENT, variable extraction "
                "and voicemail detection in realtime calls will fail"
            )

    mode = default_mode or ("pipeline" if "pipeline" in built else "realtime")
    if mode not in built:
        sys.exit(f"Cannot make '{mode}' the default: it was not built")
    default = built[mode]
    await db_client.set_default_named_model_configuration(
        organization_id, str(default.uuid)
    )
    print(f"Set '{default.name}' as the default for organization {organization_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--email", required=True, help="Email you signed up with")
    parser.add_argument(
        "--default",
        choices=["pipeline", "realtime"],
        help="Setup to make the organization default (pipeline when built)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.email, args.default))
