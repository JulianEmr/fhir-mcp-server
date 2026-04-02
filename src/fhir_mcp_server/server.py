# Copyright (c) 2025, WSO2 LLC. (https://www.wso2.com/) All Rights Reserved.

# WSO2 LLC. licenses this file to you under the Apache License,
# Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

import click
import logging

from fhir_mcp_server.utils import (
    build_user_profile,
    create_async_fhir_client,
    get_bundle_entries,
    get_default_headers,
    get_operation_outcome,
    get_operation_outcome_exception,
    get_operation_outcome_required_error,
    get_capability_statement,
    trim_resource_capabilities,
)
from fhir_mcp_server.oauth import (
    handle_failed_authentication,
    OAuthServerProvider,
    OAuthToken,
    ServerConfigs,
)
from fhirpy import AsyncFHIRClient
from fhirpy.lib import AsyncFHIRResource
from fhirpy.base.exceptions import OperationOutcome, ResourceNotFound
from fhirpy.base.searchset import Raw
from typing import Dict, Any, List
from typing_extensions import Annotated
from pydantic import AnyHttpUrl, Field
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp.server import FastMCP

logger: logging.Logger = logging.getLogger(__name__)

configs: ServerConfigs = ServerConfigs()

server_provider: OAuthServerProvider = OAuthServerProvider(configs=configs)


async def get_user_access_token() -> OAuthToken | None:
    """
    Retrieve the access token for the authenticated user.
    Returns an OAuthToken if available, otherwise raises an error.
    """
    if configs.server_access_token:
        logger.debug("Using configured FHIR access token for user.")
        return OAuthToken(
            access_token=configs.server_access_token,
            token_type="Bearer",
            client_id=configs.server_client_id,
        )

    user_token: AccessToken | None = get_access_token()
    logger.debug("Obtained client access token from context.")

    # Return the FHIR access token
    return (
        OAuthToken(
            access_token=user_token.token,
            client_id=configs.server_client_id,
            token_type="Bearer",
            expires_at=user_token.expires_at,
            scope=" ".join(user_token.scopes),
        )
        if user_token
        else None
    )


async def get_async_fhir_client() -> AsyncFHIRClient:
    """
    Get an async FHIR client with the user's access token.
    Returns an AsyncFHIRClient instance.
    """
    client_kwargs: Dict = {
        "config": configs,
        "extra_headers": get_default_headers(),
    }

    user_token: OAuthToken | None = await get_user_access_token()
    disable_auth: bool = configs.server_disable_authorization
    if not user_token:
        if not disable_auth:
            logger.error("User is not authenticated.")
            raise ValueError("User is not authenticated.")
    else:
        client_kwargs["access_token"] = user_token.access_token

    return await create_async_fhir_client(**client_kwargs)


def configure_mcp_server() -> FastMCP:
    """
    Configure and instantiate the FastMCP server instance.
    If disable_auth is True, the server will be started without authorization.
    Returns a FastMCP instance.
    """
    fastmcp_kwargs: Dict = {
        "name": "FHIR MCP Server",
        "instructions": "This server implements the HL7 FHIR MCP for secure, standards-based access to FHIR resources",
        "host": configs.mcp_host,
        "port": configs.mcp_port,
        "json_response": True,
        "stateless_http": True,
    }
    disable_auth = configs.server_disable_authorization
    if not disable_auth:
        logger.debug("Enabling authorization for FHIR MCP server.")
        auth_settings: AuthSettings = AuthSettings(
            issuer_url=AnyHttpUrl(configs.effective_server_url),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=configs.scopes,
                default_scopes=configs.scopes,
            ),
        )
        fastmcp_kwargs["auth_server_provider"] = server_provider
        fastmcp_kwargs["auth"] = auth_settings
    else:
        logger.warning("MCP authentication is disabled.")
    return FastMCP(**fastmcp_kwargs)


def register_mcp_routes(
    mcp: FastMCP,
    server_provider: OAuthServerProvider,
) -> None:
    """
    Register custom routes for the FastMCP server instance.
    """
    logger.debug("Registering custom MCP routes.")

    @mcp.custom_route("/oauth/callback", methods=["GET"])
    async def handle_auth_server_callback(request: Request) -> Response:
        """Handle MCP OAuth redirect."""
        code: str | None = request.query_params.get("code")
        state: str | None = request.query_params.get("state")

        if not code or not state:
            return handle_failed_authentication("Missing code or state parameter")

        try:
            redirect_uri: str = await server_provider.handle_mcp_oauth_callback(
                code, state
            )
            return RedirectResponse(status_code=302, url=redirect_uri)
        except Exception as ex:
            logger.error(
                "Error occurred while handling MCP oauth callback. Caused by, ",
                exc_info=ex,
            )
            return handle_failed_authentication("Something went wrong.")


def register_mcp_tools(mcp: FastMCP) -> None:
    """
    Register tool functions for the FastMCP server instance.
    """
    logger.debug("Registering MCP tools.")

    @mcp.tool(
        description=(
            "Récupère les allergies connues d'un patient donné en effectuant une interaction FHIR `search` sur la ressource AllergyIntolerance, en utilisant l'identifiant du patient comme paramètre de recherche. "
        )
    )
    async def get_allergies(
        patient_id: Annotated[
            str,
            Field(
                description=(
                    "L'identifiant du patient pour lequel récupérer les allergies. Cet identifiant est en format UUID."
                ),
                examples=["ea66758f-4c82-4b43-9171-bc803eaac8e1", "f7d82e70-e642-44ca-9d3c-6558418a60c0"],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked get_allergies with patient_id='{patient_id}'")
            if not patient_id:
                logger.error(
                    "Unable to perform search operation: 'patient_id' is a mandatory field."
                )
                return await get_operation_outcome_required_error("patient_id")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources("AllergyIntolerance").search(patient=patient_id).fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR 'AllergyIntolerance' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: 'AllergyIntolerance', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Récupère les antécédents d'un patient donné en effectuant une interaction FHIR `search` sur la ressource Condition, avec en catégorie 'problem-list-item', en utilisant l'identifiant du patient comme paramètre de recherche. "
        )
    )
    async def get_history(
        patient_id: Annotated[
            str,
            Field(
                description=(
                    "L'identifiant du patient pour lequel récupérer les antécédents. Cet identifiant est en format UUID."
                ),
                examples=["ea66758f-4c82-4b43-9171-bc803eaac8e1", "f7d82e70-e642-44ca-9d3c-6558418a60c0"],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked get_history with patient_id='{patient_id}'")
            if not patient_id:
                logger.error(
                    "Unable to perform search operation: 'patient_id' is a mandatory field."
                )
                return await get_operation_outcome_required_error("patient_id")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources("Condition").search(subject=patient_id, category="problem-list-item").fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR 'Condition' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: 'Condition', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Récupère les diagnostics d'un patient donné en effectuant une interaction FHIR `search` sur la ressource Condition, avec en catégorie 'encounter-diagnosis', en utilisant l'identifiant du patient comme paramètre de recherche. "
        )
    )
    async def get_diagnosis(
        patient_id: Annotated[
            str,
            Field(
                description=(
                    "L'identifiant du patient pour lequel récupérer les diagnostics. Cet identifiant est en format UUID."
                ),
                examples=["ea66758f-4c82-4b43-9171-bc803eaac8e1", "f7d82e70-e642-44ca-9d3c-6558418a60c0"],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked get_history with patient_id='{patient_id}'")
            if not patient_id:
                logger.error(
                    "Unable to perform search operation: 'patient_id' is a mandatory field."
                )
                return await get_operation_outcome_required_error("patient_id")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources("Condition").search(subject=patient_id, category="encounter-diagnosis").fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR 'Condition' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: 'Condition', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Récupère les résultats de laboratoire d'un patient donné en effectuant une interaction FHIR `search` sur la ressource Observation, avec en catégorie 'laboratory', en utilisant l'identifiant du patient comme paramètre de recherche. "
        )
    )
    async def get_lab_results(
        patient_id: Annotated[
            str,
            Field(
                description=(
                    "L'identifiant du patient pour lequel récupérer les résultats de laboratoire. Cet identifiant est en format UUID."
                ),
                examples=["ea66758f-4c82-4b43-9171-bc803eaac8e1", "f7d82e70-e642-44ca-9d3c-6558418a60c0"],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked get_history with patient_id='{patient_id}'")
            if not patient_id:
                logger.error(
                    "Unable to perform search operation: 'patient_id' is a mandatory field."
                )
                return await get_operation_outcome_required_error("patient_id")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources("Observation").search(subject=patient_id, category="laboratory").fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR 'Observation' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: 'Observation', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Récupère les signes vitaux d'un patient donné en effectuant une interaction FHIR `search` sur la ressource Observation, avec en catégorie 'vital-signs', en utilisant l'identifiant du patient comme paramètre de recherche. "
        )
    )
    async def get_vitals(
        patient_id: Annotated[
            str,
            Field(
                description=(
                    "L'identifiant du patient pour lequel récupérer les signes vitaux. Cet identifiant est en format UUID."
                ),
                examples=["ea66758f-4c82-4b43-9171-bc803eaac8e1", "f7d82e70-e642-44ca-9d3c-6558418a60c0"],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked get_history with patient_id='{patient_id}'")
            if not patient_id:
                logger.error(
                    "Unable to perform search operation: 'patient_id' is a mandatory field."
                )
                return await get_operation_outcome_required_error("patient_id")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources("Observation").search(subject=patient_id, category="vital-signs").fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR 'Observation' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: 'Observation', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Exécute une interaction FHIR `search` standard sur un type de ressource donné, retournant un bundle ou une liste de ressources correspondantes. "
            "Utilisez cet outil lorsque vous devez interroger plusieurs ressources sur la base d'un ou plusieurs paramètres de recherche. "
            "N'utilisez pas cet outil pour les opérations de création, de mise à jour ou de suppression, et sachez que les grands ensembles de résultats peuvent être paginés par le serveur FHIR."
        )
    )
    async def search(
        type: Annotated[
            str,
            Field(
                description="Le nom du type de ressource FHIR. Doit correspondre exactement à l'un des types de ressources pris en charge par le serveur.Le nom du type de ressource FHIR. Doit correspondre exactement à l'un des types de ressources pris en charge par le serveur.",
                examples=["MedicationRequest", "Condition", "Procedure"],
            ),
        ],
        searchParam: Annotated[
            Dict[str, str | List[str]],
            Field(
                description=(
                    "Une correspondance entre les noms des paramètres de recherche FHIR et leurs valeurs. "
                    "N'incluez que les paramètres pris en charge pour le type de ressource, comme listé par `get_capabilities`."
                ),
                examples=[
                    '{"family": "Smith"}',
                    '{"date": ["ge1970-01-01", "lt2000-01-01"]}',
                ],
            ),
        ],
    ) -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance de ressource FHIR complète correspondant aux critères de recherche."
        ),
    ]:
        try:
            logger.debug(f"Invoked with type='{type}' and searchParam={searchParam}")
            if not type:
                logger.error(
                    "Unable to perform search operation: 'type' is a mandatory field."
                )
                return await get_operation_outcome_required_error("type")

            client: AsyncFHIRClient = await get_async_fhir_client()
            async_resources: list[Any] = (
                await client.resources(type).search(Raw(**searchParam)).fetch_raw()
            )
            logger.debug("Async resources fetched:", async_resources) 
            return async_resources
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR '{type}' resource search operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform search operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while searching the resource: '{type}', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR search operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Effectue une interaction FHIR read pour récupérer une instance unique de ressource via son type et son identifiant (resource ID), en affinant éventuellement la réponse avec des paramètres de recherche ou des opérations personnalisées."
            "Utilisez cet outil lorsque vous connaissez l'identifiant exact de la ressource et que vous ne requérez que cette ressource spécifique ; ne l'utilisez pas pour des requêtes de masse (bulk)."
            "Si des paramètres de requête additionnels ou des opérations au niveau de l'instance sont nécessaires (par exemple, _elements ou $validate), incluez-les dans les champs searchParam ou operation."
        )
    )
    async def read(
        type: Annotated[
            str,
            Field(
                description="Le nom du type de ressource FHIR. Doit correspondre exactement à l'un des types de ressources pris en charge par le serveur.",
                examples=["DiagnosticReport", "AllergyIntolerance", "Immunization"],
            ),
        ],
        id: Annotated[
            str,
            Field(description="Le ID logique d'une instance spécifique de ressource FHIR."),
        ],
        searchParam: Annotated[
            Dict[str, str | List[str]],
            Field(
                description=(
                    "Une correspondance entre les noms des paramètres de recherche FHIR et leurs valeurs. "
                    "N'incluez que les paramètres pris en charge pour le type de ressource, comme listé par `get_capabilities`."
                ),
                examples=['{"device-name": "glucometer", "identifier": ["12345"]}'],
            ),
        ] = {},
        operation: Annotated[
            str,
            Field(
                description=(
                    "Le nom d'une opération FHIR personnalisée ou d'une requête étendue définie pour la ressource "
                    "doit correspondre à l'un des noms d'opération retournés par `get_capabilities`."
                ),
                examples=["$everything"],
            ),
        ] = "",
    ) -> Annotated[
        Dict[str, Any],
        Field(
            description="Un dictionnaire contenant l'instance unique de la ressource FHIR correspondant au type et à l'identifiant demandés."
        ),
    ]:
        try:
            logger.debug(
                f"Invoked with type='{type}', id={id}, searchParam={searchParam}, and operation={operation}"
            )
            if not type:
                logger.error(
                    "Unable to perform read operation: 'type' is a mandatory field."
                )
                return await get_operation_outcome_required_error("type")

            client: AsyncFHIRClient = await get_async_fhir_client()
            bundle: dict = await client.resource(resource_type=type, id=id).execute(
                operation=operation or "", method="GET", params=searchParam
            )

            return await get_bundle_entries(bundle=bundle)
        except ResourceNotFound as ex:
            logger.error(
                f"Resource of type '{type}' with id '{id}' not found. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="not-found",
                diagnostics=f"The resource of type '{type}' with id '{id}' was not found.",
            )
        except ValueError as ex:
            logger.exception(
                f"User does not have permission to perform FHIR '{type}' resource read operation. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics=f"The user does not have the rights to perform read operation.",
            )
        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server returned an OperationOutcome error while reading the resource: '{type}', Caused by,",
                exc_info=ex,
            )
            return ex.resource["issue"] or await get_operation_outcome_exception()
        except Exception as ex:
            logger.exception(
                f"An unexpected error occurred during the FHIR read operation for resource: '{type}'. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()

    @mcp.tool(
        description=(
            "Récupère le profile FHIR de l'utilisateur authentifié en effectuant une interaction FHIR `read` sur la ressource identifiée par le token d'accès de l'utilisateur. "
            "Utilisez cet outil pour obtenir les informations démographiques de base de l'utilisateur (telles que 'id', 'name', 'birthDate') à partir de son profil FHIR, en supposant que le token d'accès contient les références nécessaires pour localiser la ressource utilisateur correspondante sur le serveur FHIR. "
        )
    )
    async def get_user() -> Annotated[
        list[Dict[str, Any]] | Dict[str, Any],
        Field(
            description="Un dictionnaire contenant les informations démographiques de l'utilisateur authentifié, telles que 'id', 'name', et 'birthDate'."
        ),
    ]:
        try:
            logger.debug("Retrieving authenticated user's profile.")

            # Validate user authentication
            user_token = await get_user_access_token()
            if not user_token:
                logger.debug("Unauthorized access attempt to get_me endpoint.")
                return {}

            # Retrieve token metadata
            token_metadata = server_provider.token_metadata_mapping.get(
                user_token.access_token
            )
            if not token_metadata:
                logger.debug("Token metadata not found for authenticated user.")
                return {}

            # Extract ID token information
            id_token = token_metadata.get_id_token()
            if not id_token:
                logger.debug("ID token not found in token metadata.")
                return {}

            # Validate resource identifiers
            resource_id = id_token.resource_id
            resource_type = id_token.resource_type

            if not resource_id or not resource_type:
                logger.debug("Resource ID or type missing from ID token.")
                return {}

            logger.debug(f"Fetching FHIR resource: {resource_type}/{resource_id}")

            # Fetch user's FHIR resource
            client: AsyncFHIRClient = await get_async_fhir_client()
            resource: Dict[str, Any] = await client.get(
                resource_type_or_resource_or_ref=resource_type, id_or_ref=resource_id
            )

            # Build response with only available fields
            profile: Dict[str, Any] = build_user_profile(resource)

            logger.debug(
                f"Successfully retrieved profile for user: {resource_type}/{resource_id}"
            )
            return profile

        except ValueError as ex:
            logger.exception(
                "Authorization error occurred while reading user resource. Caused by, ",
                exc_info=ex,
            )
            return await get_operation_outcome(
                code="forbidden",
                diagnostics="The user does not have the rights to perform read operations.",
            )

        except OperationOutcome as ex:
            logger.exception(
                f"FHIR server error occurred while reading user resource. Caused by, ",
                exc_info=ex,
            )
            return ex.resource.get("issue") or await get_operation_outcome_exception()

        except Exception as ex:
            logger.exception(
                "Unexpected error occurred while reading user resource. Caused by, ",
                exc_info=ex,
            )
        return await get_operation_outcome_exception()


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="streamable-http",
    show_default=True,
    help="Transport protocol to use",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Log level to use",
)
def main(transport, log_level) -> int:
    """
    FHIR MCP Server - helping you expose any FHIR Server or API as a MCP Server.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="[%(asctime)s] %(levelname)s {%(name)s.%(funcName)s:%(lineno)d} - %(message)s",
    )
    try:
        mcp: FastMCP = configure_mcp_server()
        register_mcp_tools(mcp=mcp)
        register_mcp_routes(mcp=mcp, server_provider=server_provider)
        logger.info(f"Starting FHIR MCP server with {transport} transport")
        mcp.run(transport=transport)
    except Exception as ex:
        logger.error(
            f"Unable to run the FHIR MCP server. Caused by, %s", ex, exc_info=True
        )
        return 1
    return 0
