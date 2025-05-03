# MCP Azure Storage Sample

This repository contains sample projects demonstrating how to use the Model Context Protocol (MCP) with Azure Blob Storage. It includes both a server and a client implementation.

## Project Overview

- **MCP Server for Azure Blob Storage**  
  Implements an MCP server that exposes Azure Blob Storage operations (list, create, delete containers/blobs, upload/download blobs) via asynchronous Python APIs.  
  - Uses Microsoft Entra Managed Identity or Azure CLI authentication.
  - Designed for integration with MCP-compatible applications.
  - [Read the full server README](./mcp-server-az-storage-svc/README.md)

- **MCP Client AI Assistant**  
  Companion client application that interacts with the MCP server to perform storage operations.  
  - Implements MCP client functionality with an AI-powered chat interface using Azure OpenAI GPT-4o.
  - Discovers available tools from the server and allows natural language interaction with Azure Blob Storage.
  - [Read the full client README](./mcp-client-ai-assistant/README.md)

## Quick Links

- [MCP Server README](./mcp-server-az-storage-svc/README.md)
- [MCP Client README](./mcp-client-ai-assistant/README.md)

For setup, usage, and detailed features, please refer to the respective README files above.
