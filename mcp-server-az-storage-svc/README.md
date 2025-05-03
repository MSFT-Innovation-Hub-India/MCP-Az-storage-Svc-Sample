# MCP Server for Azure Blob Storage

This repository implements an MCP (Model Context Protocol) server that exposes various functionalities on an Azure Blob Storage account. It is based on the MCP SDK from Anthropic, which can be found at: https://github.com/modelcontextprotocol/python-sdk

The service provides asynchronous Python APIs for managing containers and blobs, making it easy to integrate Azure Storage into your MCP-compatible applications.
It uses Microsoft Entra Managed Identity to access the Storage Account in Azure.

If this app is run on a terminal locally, the user must perform an az login and be able to access the Storage Account to perform the operations listed below
If run in Azure Container App, enable System Identity on the service, and assign it necessary permissions on the Storage Account (For this demo, I assigned the roles Storage Account Contributor, Storage Blob Data owner, Storage Blob Data Reader, etc.)

After you run server, you can use the companion app, mcp-client-ai-assistant to invoke it and perform the different actions on the storage account. The latter implements the mcp client functionality

## Project Structure

```
mcp-server-az-storage-svc
├── Dockerfile                # Docker image build instructions
├── README.md                 # Project documentation (this file)
└── mcp-server
    ├── server.py             # Main application entry point
    ├── az_storage_svcs.py    # AzureBlobStorageService implementation
    ├── requirements.txt      # Python dependencies
    └── __pycache__/          # Compiled Python files
```

## Features

- **List Containers**: Retrieve all blob containers in the storage account.
- **Create Container**: Create a new blob container.
- **Delete Container**: Remove an existing blob container.
- **List Blobs**: List all blobs within a specified container.
- **Upload Blob**: Upload content to a specified blob.
- **Download Blob**: Download content from a specified blob.
- **Delete Blob**: Remove a specified blob from a container.

## Prerequisites

- Python 3.7 or higher
- Azure account with Blob Storage access
- Docker (optional, for containerization)

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd mcp-server-az-storage-svc
   ```
2. Install dependencies:
   ```sh
   pip install -r mcp-server/requirements.txt
   ```

## Usage

### Running Locally

Run the server application:
```sh
python mcp-server/server.py
```
Set the Azure Storage Account name in the .env file.

### Docker

Build the Docker image:
```sh
docker build -t azure-blob-storage-service .
```

Run the container:
```sh
docker run -p 8000:8000 azure-blob-storage-service
```

The service will be available at `http://localhost:8000`.

## Configuration

Set your Azure Storage connection string as an environment variable before running the service:
```sh
export AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>"
```
On Windows (Command Prompt):
```cmd
set AZURE_STORAGE_CONNECTION_STRING=<your-connection-string>
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.