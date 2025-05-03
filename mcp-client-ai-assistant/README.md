# MCP Client for Azure Blob Storage

## Overview

This application is a Model Context Protocol (MCP) client that connects to the companion MCP server using Server-Sent Events (SSE) protocol. The client acts as an AI-powered chat interface, allowing users to interact with Azure Blob Storage services through natural language commands.

## Features

- **AI-Powered Interface**: Uses Azure OpenAI's GPT-4o model to understand and process natural language requests
- **Tool Discovery**: Automatically discovers available tools and actions from the MCP server
- **Azure Blob Storage Integration**: Performs actions on Azure Blob Storage through the MCP server
- **Resilient Connections**: Implements connection retries, exponential backoff, and proper resource management.
The Storage Account itself is in the mcp server configuration, for this demo. The mcp client does not specify that.

## How It Works

1. The client establishes a connection to the MCP server via SSE protocol
2. It discovers available tools and their capabilities from the server
3. When users enter natural language queries, the GPT-4o model:
   - Interprets the user's intent
   - Selects the appropriate tool to accomplish the task
   - Formats requests to the MCP server
   - Translates raw responses into natural language

## Architecture

The client consists of several key components:
- `ServerConnection`: Manages SSE communication with the MCP server with proper resource lifecycle
- `Tool`: Represents discovered capabilities from the server with validation logic
- `AzureOpenAIClient`: Handles communication with Azure OpenAI using managed identity
- `ChatSession`: Coordinates between user input, LLM, and tool execution

## Requirements

- Python 3.8 or higher
- Azure OpenAI access with GPT-4o deployment
- Azure Storage Account (connected to the companion MCP server)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mcp-client-ai-assistant

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Configure environment variables in `.env`:

```properties
# MCP Server Configuration
MCP_SERVER_URL=https://your-mcp-server-url.azurecontainerapps.io/sse

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# AZURE_OPENAI_API_KEY=your-api-key-here  # Only needed if not using managed identity
```

## Usage

Run the client:
```bash
python client.py
```

The chat interface will start, allowing you to:
- Type natural language requests to interact with Azure Blob Storage
- View available tools by typing `tools`
- Exit the application by typing `exit` or `quit`

## Example Interactions

- "List all containers in my storage account"
- "Upload file.txt to the documents container"
- "Download the latest invoice from the invoices container"
- "Generate a SAS token for the images container"

## Security

This client follows Azure best practices for security:
- Uses Azure Managed Identity for authentication when available
- Implements proper error handling with retries and exponential backoff
- Ensures secure connection management and resource cleanup
- Validates all tool inputs before execution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[LICENSE INFORMATION]