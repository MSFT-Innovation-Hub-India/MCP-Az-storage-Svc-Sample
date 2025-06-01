"""
Azure Blob Storage service for MCP Server.
This module provides access to Azure Blob Storage functionality through the MCP protocol.
"""

import os
import json
import base64
import logging
from typing import Dict, List, Any, Optional, Union
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    ContentSettings,
)
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from azure.core.pipeline.transport import RequestsTransport
from azure.core.pipeline.policies import RetryPolicy

logger = logging.getLogger(__name__)


class AzureBlobStorageService:
    """
    Azure Blob Storage service provider for MCP.

    This service allows chat models to access and manage Azure Blob Storage
    resources through the MCP protocol.
    """

    def __init__(self, storage_account_name: Optional[str] = None):
        """
        Initialize the Azure Blob Storage service using DefaultAzureCredential.

        Args:
            storage_account_name (Optional[str]): Azure Storage account name.
                If not provided, will look for AZURE_STORAGE_ACCOUNT env var.
        """
        # Use provided storage account name or get from environment using dotenv
        self._storage_account_name = storage_account_name or os.getenv(
            "AZURE_STORAGE_ACCOUNT"
        )

        if not self._storage_account_name:
            logger.warning(
                "No Azure Storage account name provided in parameters or .env file. "
                "Service will be initialized but not functional."
            )
            self._blob_service_client = None
        else:
            # Initialize the Blob Service Client with DefaultAzureCredential
            try:
                # Create the connection string
                account_url = (
                    f"https://{self._storage_account_name}.blob.core.windows.net"
                )                # Create the BlobServiceClient using the storage account key
                
                # Create the BlobServiceClient using the account key
                # Note: In newer Azure SDK versions, retry policies are configured differently
                self._blob_service_client = BlobServiceClient(
                    account_url=account_url, credential=DefaultAzureCredential()
                )

                # Configure retry policy on the client's transport pipeline
                self._blob_service_client._client._config.retry_policy = RetryPolicy(
                    retry_total=3, retry_mode="exponential", retry_backoff_factor=1
                )                # Test the connection to catch issues early
                self._blob_service_client.get_service_properties()
                logger.info(
                    f"Azure Blob Storage service initialized successfully using DefaultAzureCredential for account: {self._storage_account_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Storage: {str(e)}")
                self._blob_service_client = None

    @property
    def service_name(self) -> str:
        """
        Get the name of this service.

        Returns:
            str: The service name
        """
        return "azure_blob_storage"

    async def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get the list of tools provided by this service.

        Returns:
            List[Dict[str, Any]]: List of tool definitions following the MCP specification
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_containers",
                    "description": "List all blob containers in the storage account",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_container",
                    "description": "Create a new blob container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to create",
                            }
                        },
                        "required": ["container_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_container",
                    "description": "Delete a blob container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to delete",
                            }
                        },
                        "required": ["container_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_blobs",
                    "description": "List all blobs in a container",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to list blobs from",
                            },
                            "prefix": {
                                "type": "string",
                                "description": "Optional prefix to filter blobs by name",
                            },
                        },
                        "required": ["container_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "upload_blob",
                    "description": "Upload content to a blob",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to upload to",
                            },
                            "blob_name": {
                                "type": "string",
                                "description": "Name of the blob to create or overwrite",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to upload to the blob",
                            },
                            "content_type": {
                                "type": "string",
                                "description": "Content type of the blob (e.g., 'text/plain', 'application/json')",
                            },
                        },
                        "required": ["container_name", "blob_name", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "download_blob",
                    "description": "Download content from a blob",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container",
                            },
                            "blob_name": {
                                "type": "string",
                                "description": "Name of the blob to download",
                            },
                        },
                        "required": ["container_name", "blob_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_blob",
                    "description": "Delete a blob",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container",
                            },
                            "blob_name": {
                                "type": "string",
                                "description": "Name of the blob to delete",
                            },
                        },
                        "required": ["container_name", "blob_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "upload_image_blob",
                    "description": "Upload a base64-encoded image to a blob. Content type defaults to image/png.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to upload to",
                            },
                            "blob_name": {
                                "type": "string",
                                "description": "Name of the blob to create or overwrite",
                            },
                            "image_base64": {
                                "type": "string",
                                "description": "Base64-encoded image data",
                            },
                            "content_type": {
                                "type": "string",
                                "description": "Content type of the image (e.g., 'image/png', 'image/jpeg'). Defaults to 'image/png'",
                                "default": "image/png",
                            },
                        },
                        "required": ["container_name", "blob_name", "image_base64"],
                    },
                },
            },
        ]

    async def get_actions(self) -> List[Dict[str, Any]]:
        """
        Get the list of actions provided by this service.

        Returns:
            List[Dict[str, Any]]: List of action definitions following the MCP specification
        """
        # This service doesn't provide any actions
        return []

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a tool based on its name and arguments.

        Args:
            tool_name (str): The name of the tool to execute
            tool_args (Dict[str, Any]): Arguments for the tool

        Returns:
            Any: The result of the tool execution

        Raises:
            ValueError: If the tool is not supported or parameters are invalid
            RuntimeError: If the service is not properly initialized
        """
        if not self._blob_service_client:
            raise RuntimeError("Azure Blob Storage service is not properly initialized")

        try:
            if tool_name == "list_containers":
                return await self._list_containers()

            elif tool_name == "create_container":
                container_name = tool_args.get("container_name")
                if not container_name:
                    raise ValueError("container_name is required")
                return await self._create_container(container_name)

            elif tool_name == "delete_container":
                container_name = tool_args.get("container_name")
                if not container_name:
                    raise ValueError("container_name is required")
                return await self._delete_container(container_name)

            elif tool_name == "list_blobs":
                container_name = tool_args.get("container_name")
                prefix = tool_args.get("prefix", "")
                if not container_name:
                    raise ValueError("container_name is required")
                return await self._list_blobs(container_name, prefix)

            elif tool_name == "upload_blob":
                container_name = tool_args.get("container_name")
                blob_name = tool_args.get("blob_name")
                content = tool_args.get("content")
                content_type = tool_args.get("content_type", "application/octet-stream")

                if not container_name or not blob_name or content is None:
                    raise ValueError(
                        "container_name, blob_name, and content are required"
                    )

                return await self._upload_blob(
                    container_name, blob_name, content, content_type
                )

            elif tool_name == "download_blob":
                container_name = tool_args.get("container_name")
                blob_name = tool_args.get("blob_name")

                if not container_name or not blob_name:
                    raise ValueError("container_name and blob_name are required")

                return await self._download_blob(container_name, blob_name)

            elif tool_name == "delete_blob":
                container_name = tool_args.get("container_name")
                blob_name = tool_args.get("blob_name")

                if not container_name or not blob_name:
                    raise ValueError("container_name and blob_name are required")

                return await self._delete_blob(container_name, blob_name)

            else:
                raise ValueError(f"Tool '{tool_name}' is not supported by this service")

        except ResourceNotFoundError as e:
            error_msg = f"Resource not found: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

        except ResourceExistsError as e:
            error_msg = f"Resource already exists: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

        except AzureError as e:
            error_msg = f"Azure error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def _list_containers(self) -> Dict[str, Any]:
        """List all containers in the storage account"""
        try:
            containers = []
            container_list = self._blob_service_client.list_containers(
                include_metadata=True
            )

            for container in container_list:
                containers.append(
                    {
                        "name": container.name,
                        "last_modified": (
                            container.last_modified.isoformat()
                            if container.last_modified
                            else None
                        ),
                        "metadata": container.metadata,
                    }
                )

            return {"containers": containers}
        except Exception as e:
            logger.error(f"Error listing containers: {str(e)}")
            raise

    async def _create_container(self, container_name: str) -> Dict[str, Any]:
        """Create a new container"""
        try:
            container_client = self._blob_service_client.create_container(
                container_name
            )
            return {"success": True, "name": container_name, "created": True}
        except ResourceExistsError:
            return {
                "success": False,
                "name": container_name,
                "error": "Container already exists",
            }
        except Exception as e:
            logger.error(f"Error creating container {container_name}: {str(e)}")
            raise

    async def _delete_container(self, container_name: str) -> Dict[str, Any]:
        """Delete a container"""
        try:
            self._blob_service_client.delete_container(container_name)
            return {"success": True, "name": container_name, "deleted": True}
        except ResourceNotFoundError:
            return {
                "success": False,
                "name": container_name,
                "error": "Container not found",
            }
        except Exception as e:
            logger.error(f"Error deleting container {container_name}: {str(e)}")
            raise

    async def _list_blobs(
        self, container_name: str, prefix: str = ""
    ) -> Dict[str, Any]:
        """List blobs in a container with optional prefix filter"""
        try:
            container_client = self._blob_service_client.get_container_client(
                container_name
            )
            blobs = []
            blob_list = container_client.list_blobs(name_starts_with=prefix)

            for blob in blob_list:
                blobs.append(
                    {
                        "name": blob.name,
                        "size": blob.size,
                        "content_type": blob.content_settings.content_type,
                        "last_modified": (
                            blob.last_modified.isoformat()
                            if blob.last_modified
                            else None
                        ),
                        "metadata": blob.metadata,
                    }
                )

            return {"container_name": container_name, "blobs": blobs}
        except ResourceNotFoundError:
            return {
                "success": False,
                "container_name": container_name,
                "error": "Container not found",
            }
        except Exception as e:
            logger.error(f"Error listing blobs in container {container_name}: {str(e)}")
            raise

    async def _upload_blob(
        self, container_name: str, blob_name: str, content: str, content_type: str
    ) -> Dict[str, Any]:
        """Upload content to a blob"""
        try:
            # Ensure container exists
            try:
                self._blob_service_client.create_container(container_name)
                logger.info(f"Container {container_name} created")
            except ResourceExistsError:
                logger.info(f"Container {container_name} already exists")

            # Get blob client and upload the content
            blob_client = self._blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            content_settings = ContentSettings(content_type=content_type)

            # Convert content to bytes
            if isinstance(content, str):
                data = content.encode("utf-8")
            else:
                data = content

            # Upload the blob with optimized settings
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=content_settings,
                metadata={"uploaded": "by_mcp_service"},
            )

            return {
                "success": True,
                "container_name": container_name,
                "blob_name": blob_name,
                "content_type": content_type,
                "size": len(data),
                "url": blob_client.url,
            }
        except Exception as e:
            logger.error(f"Error uploading blob {blob_name}: {str(e)}")
            raise

    async def _download_blob(
        self, container_name: str, blob_name: str
    ) -> Dict[str, Any]:
        """Download content from a blob"""
        try:
            blob_client = self._blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            download = blob_client.download_blob()
            content = await download.readall()

            # Try to decode as text if possible
            try:
                content_str = content.decode("utf-8")
                is_text = True
            except UnicodeDecodeError:
                # If it's not text, encode as base64
                content_str = base64.b64encode(content).decode("ascii")
                is_text = False

            return {
                "success": True,
                "container_name": container_name,
                "blob_name": blob_name,
                "content": content_str,
                "content_type": download.properties.content_settings.content_type,
                "size": download.properties.size,
                "is_text": is_text,
                "metadata": download.properties.metadata,
            }
        except ResourceNotFoundError:
            return {
                "success": False,
                "container_name": container_name,
                "blob_name": blob_name,
                "error": "Blob not found",
            }
        except Exception as e:
            logger.error(f"Error downloading blob {blob_name}: {str(e)}")
            raise

    async def _delete_blob(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Delete a blob"""
        try:
            blob_client = self._blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            blob_client.delete_blob()

            return {
                "success": True,
                "container_name": container_name,
                "blob_name": blob_name,
                "deleted": True,
            }
        except ResourceNotFoundError:
            return {
                "success": False,
                "container_name": container_name,
                "blob_name": blob_name,
                "error": "Blob not found",
            }
        except Exception as e:
            logger.error(f"Error deleting blob {blob_name}: {str(e)}")
            raise    
    
    async def _upload_image_blob(
        self, container_name: str, blob_name: str, image_base64: str, content_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """Upload a base64-encoded image to a blob"""
        
        logger.info("the received image")
        logger.info(image_base64)
        
        try:
            # Ensure container exists
            try:
                logger.info(f">>>>> In upload image_blob: Ensuring container {container_name} exists")
                self._blob_service_client.create_container(container_name)
                logger.info(f"Container {container_name} created")
            except ResourceExistsError:
                logger.info(f"Container {container_name} already exists")

            # Decode the base64 string
            binary_data = base64.b64decode(image_base64)
            # Get blob client and upload the image
            blob_client = self._blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            content_settings = ContentSettings(content_type=content_type)
            blob_client.upload_blob(
                binary_data,
                blob_type="BlockBlob"
            )
            return {
                "success": True,
                "container_name": container_name,
                "blob_name": blob_name,
                "url": blob_client.url
            }
        except ResourceExistsError as e:
            # This should be caught by the earlier try/except, but just in case
            logger.warning(f"Container {container_name} already exists: {str(e)}")
            # Continue with the upload despite this error
        except Exception as e:
            logger.error(f"Error uploading image blob {blob_name}: {str(e)}")
            return {
                "success": False,
                "container_name": container_name,
                "blob_name": blob_name,
                "error": str(e)
            }