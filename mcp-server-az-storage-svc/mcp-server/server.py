from mcp.server.fastmcp import FastMCP
# Import the Azure Blob Storage Service
from az_storage_svcs import AzureBlobStorageService

# Initialize FastMCP with transport_settings
mcp = FastMCP("Contoso-Blob Storage-Services")

# Initialize the Azure Blob Storage Service
blob_service = AzureBlobStorageService()

# Register basic resources and tools
@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    return f"Resource echo: {message}"

@mcp.tool()
def echo_tool(message: str) -> str:
    return f"Tool echo: {message}"

@mcp.prompt()
def echo_prompt(message: str) -> str:
    return f"Please process this message: {message}"

# Register Azure Blob Storage tools using FastMCP's tool decorators
@mcp.tool()
async def list_containers() -> dict:
    """List all blob containers in the storage account"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._list_containers()

@mcp.tool()
async def create_container(container_name: str) -> dict:
    """Create a new blob container"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._create_container(container_name)

@mcp.tool()
async def delete_container(container_name: str) -> dict:
    """Delete a blob container"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._delete_container(container_name)

@mcp.tool()
async def list_blobs(container_name: str, prefix: str = "") -> dict:
    """List all blobs in a container"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._list_blobs(container_name, prefix)

@mcp.tool()
async def upload_blob(container_name: str, blob_name: str, content: str, content_type: str = "application/octet-stream") -> dict:
    """Upload content to a blob"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._upload_blob(container_name, blob_name, content, content_type)

@mcp.tool()
async def download_blob(container_name: str, blob_name: str) -> dict:
    """Download content from a blob"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._download_blob(container_name, blob_name)

@mcp.tool()
async def delete_blob(container_name: str, blob_name: str) -> dict:
    """Delete a blob"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._delete_blob(container_name, blob_name)

@mcp.tool()
async def upload_image_blob(container_name: str, blob_name: str, image_base64: str, content_type: str = "image/jpeg") -> dict:
    """Upload a base64-encoded image to a blob"""
    if not blob_service._blob_service_client:
        return {"error": "Storage service not initialized"}
    return await blob_service._upload_image_blob(container_name, blob_name, image_base64, content_type)

if __name__ == "__main__":
    import os
    import uvicorn
    
    # Get the FastAPI app with SSE transport
    app = mcp.sse_app()
    
    # Get port from environment with default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run using Uvicorn server, binding to 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=port)
