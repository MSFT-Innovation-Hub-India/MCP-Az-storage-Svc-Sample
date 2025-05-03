#!/usr/bin/env python
"""
MCP Client implementation for interacting with the MCP server.
This client follows Azure best practices for resource management, error handling,
and secure connection handling.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple, Union
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import httpx
from azure.identity import DefaultAzureCredential
from azure.core.credentials import TokenCredential
try:
    # Python SDK Version 1.0.0 and above
    from openai import AzureOpenAI
except ImportError:
    # Use older Azure OpenAI package if available
    try:
        from azure.ai.openai import AzureOpenAI
    except ImportError:
        raise ImportError("Neither the openai package (v1.0.0+) nor azure.ai.openai package is installed")

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Configure logging with ISO format timestamps for better traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        
        # Azure OpenAI configuration
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-sansri-eastus2.openai.azure.com/")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")  # Only used as fallback if managed identity fails
        
        # MCP Server URL - Read from environment variable with fallback
        self.mcp_server_url = os.getenv("MCP_SERVER_URL")
        if not self.mcp_server_url:
            self.mcp_server_url = "http://localhost:8000/sse"  # Only used as fallback
            logger.warning(f"MCP_SERVER_URL not found in environment, using default: {self.mcp_server_url}")
        else:
            logger.info(f"Using MCP server URL from environment: {self.mcp_server_url}")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise

    @property
    def azure_credential(self) -> TokenCredential:
        """
        Get Azure managed identity credential.
        
        Returns:
            DefaultAzureCredential
        """
        try:
            return DefaultAzureCredential()
        except Exception as e:
            logger.error(f"Error creating Azure credential: {e}")
            raise ValueError(f"Failed to create Azure managed identity credential: {e}")


class ServerConnection:
    """
    Manages connection to an MCP server with proper resource lifecycle management.
    Implements robust error handling and timeout management.
    """
    
    def __init__(self, server_url: str) -> None:
        """
        Initialize the ServerConnection with the server URL.
        
        Args:
            server_url: The URL of the MCP server, e.g., http://localhost:8000/sse
        """
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self._cleanup_lock = asyncio.Lock()
        self.exit_stack = AsyncExitStack()
        self._tools_cache: Optional[List[Any]] = None
        self.connected = False
    
    async def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to the MCP server with timeout.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection succeeded, False otherwise
        """
        try:
            # Create connection with timeout
            connection_task = self._connect()
            await asyncio.wait_for(connection_task, timeout=timeout)
            self.connected = True
            return True
        except asyncio.TimeoutError:
            logger.error(f"Connection to {self.server_url} timed out after {timeout}s")
            await self.cleanup()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_url}: {e}")
            await self.cleanup()
            return False
    
    async def _connect(self) -> None:
        """
        Internal method to establish server connection with proper resource tracking.
        """
        try:
            # Connect to the server using SSE
            read_write = await self.exit_stack.enter_async_context(sse_client(self.server_url))
            read_stream, write_stream = read_write
            
            # Create and initialize session
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self.session = session
            logger.info(f"Connected to MCP server at {self.server_url}")
            
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            await self.cleanup()
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of tool definitions
            
        Raises:
            RuntimeError: If not connected to server
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        if self._tools_cache is None:
            tools_response = await self.session.list_tools()
            self._tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } 
                for tool in tools_response.tools
            ]
            
        return self._tools_cache
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server with retry logic.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            retries: Number of retry attempts for transient failures
            retry_delay: Base delay between retries (will use exponential backoff)
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If not connected to server
            ValueError: If the tool doesn't exist
            Exception: On tool execution failure after all retries
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        # Verify tool exists
        tools = await self.list_tools()
        if not any(tool["name"] == tool_name for tool in tools):
            raise ValueError(f"Tool '{tool_name}' not found on MCP server")
        
        # Implement retry with exponential backoff
        attempt = 0
        last_exception = None
        
        while attempt <= retries:
            try:
                logger.info(f"Executing tool '{tool_name}' (attempt {attempt + 1}/{retries + 1})...")
                result = await self.session.call_tool(tool_name, arguments)
                
                if result and result.content:
                    try:
                        return json.loads(result.content[0].text)
                    except json.JSONDecodeError:
                        return {"text": result.content[0].text}
                else:
                    return {"error": "No content received from tool"}
                    
            except Exception as e:
                last_exception = e
                attempt += 1
                if attempt <= retries:
                    delay = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"Tool execution failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool execution failed after {retries + 1} attempts: {e}")
                    raise Exception(f"Failed to execute tool '{tool_name}': {e}") from last_exception
    
    async def cleanup(self) -> None:
        """
        Clean up all resources safely. Can be called multiple times.
        """
        async with self._cleanup_lock:
            logger.debug("Cleaning up server connection resources")
            try:
                await self.exit_stack.aclose()
                self.session = None
                self._tools_cache = None
                self.connected = False
                logger.info("Server connection resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during resource cleanup: {e}")


class Tool:
    """
    Represents an MCP tool with rich metadata and utility methods.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any]
    ) -> None:
        """
        Initialize a Tool object.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for tool input
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._required_params = input_schema.get("required", [])
        self._properties = input_schema.get("properties", {})
    
    @property
    def required_params(self) -> List[str]:
        """Get list of required parameters."""
        return self._required_params
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get tool parameter definitions."""
        return self._properties
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate arguments against tool schema.
        
        Args:
            arguments: Arguments to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        for param in self.required_params:
            if param not in arguments:
                return False, f"Required parameter '{param}' is missing"
        
        # Validate parameter types (basic validation only)
        for param_name, param_value in arguments.items():
            if param_name in self._properties:
                param_schema = self._properties[param_name]
                param_type = param_schema.get("type")
                
                # Very basic type checking
                if param_type == "string" and not isinstance(param_value, str):
                    return False, f"Parameter '{param_name}' should be a string"
                elif param_type == "number" and not isinstance(param_value, (int, float)):
                    return False, f"Parameter '{param_name}' should be a number"
                elif param_type == "integer" and not isinstance(param_value, int):
                    return False, f"Parameter '{param_name}' should be an integer"
                elif param_type == "boolean" and not isinstance(param_value, bool):
                    return False, f"Parameter '{param_name}' should be a boolean"
                elif param_type == "array" and not isinstance(param_value, list):
                    return False, f"Parameter '{param_name}' should be an array"
                elif param_type == "object" and not isinstance(param_value, dict):
                    return False, f"Parameter '{param_name}' should be an object"
        
        return True, None
    
    def format_for_display(self) -> str:
        """
        Format tool information for display.
        
        Returns:
            A formatted string describing the tool
        """
        param_descriptions = []
        for param_name, param_info in self._properties.items():
            required = " (required)" if param_name in self._required_params else ""
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "No description")
            param_descriptions.append(f"  - {param_name}: {param_desc} (type: {param_type}){required}")
        
        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n" + "\n".join(param_descriptions)
        )


class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI using managed identity or API key authentication.
    Follows Azure best practices for authentication.
    """
    
    def __init__(self, credential: TokenCredential, endpoint: str, deployment: str, api_key: Optional[str] = None) -> None:
        """
        Initialize the Azure OpenAI client.
        
        Args:
            credential: Azure credential for authentication
            endpoint: Azure OpenAI endpoint URL
            deployment: Deployment name of the model
            api_key: Optional API key as fallback
        """
        self.deployment = deployment
        self.endpoint = endpoint
        
        # First try to use Azure AD authentication
        try:
            # Get token from credential
            # token = credential.get_token("https://cognitiveservices.azure.com/.default")
              # Initialize Azure OpenAI Service client with Entra ID authentication
            token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
            
            # The AzureOpenAI client implementation changed in the OpenAI Python SDK v1.0.0
            try:
                # Newer OpenAI package approach
                self.client = AzureOpenAI(
                    azure_ad_token_provider=token_provider,  # Use token directly as api_key
                    azure_endpoint=endpoint,
                    api_version="2023-12-01-preview"
                )
                logger.info("Initialized Azure OpenAI client with managed identity authentication")
            except (TypeError, ValueError):
                if api_key:
                    # Fall back to API key if token approach doesn't work
                    self.client = AzureOpenAI(
                        azure_ad_token_provider=token_provider,
                        azure_endpoint=endpoint,
                        api_version="2023-12-01-preview"
                    )
                    logger.info("Initialized Azure OpenAI client with API key authentication")
                else:
                    raise ValueError("Unable to authenticate with managed identity and no API key provided")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> str:
        """
        Get a completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            The LLM's response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            # Azure OpenAI client is synchronous, so use loop.run_in_executor to make it async
            loop = asyncio.get_event_loop()
            
            # Handle compatibility with different client versions
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1
                )
            )
            
            # Handle different response formats
            try:
                return response.choices[0].message.content
            except AttributeError:
                # Older client might have different response structure
                return response.choices[0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            raise Exception(f"Azure OpenAI request failed: {str(e)}")
    
    async def close(self) -> None:
        """Close the client (placeholder for interface compatibility)."""
        # Azure OpenAI client doesn't need explicit closing
        pass


class ChatSession:
    """
    Main chat session handler coordinating LLM and MCP server interactions.
    """
    
    def __init__(
        self, 
        server_connection: ServerConnection,
        llm_client: AzureOpenAIClient
    ) -> None:
        """
        Initialize a chat session.
        
        Args:
            server_connection: Connection to MCP server
            llm_client: LLM client for completions
        """
        self.server = server_connection
        self.llm_client = llm_client
        self.messages: List[Dict[str, str]] = []
        self.tools: List[Tool] = []
    
    async def initialize(self) -> bool:
        """
        Initialize the chat session.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            # Connect to server
            if not await self.server.connect():
                return False
            
            # Fetch available tools
            tool_data = await self.server.list_tools()
            self.tools = [
                Tool(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["input_schema"]
                )
                for tool in tool_data
            ]
            
            # Set up system message with tools
            tools_description = "\n\n".join([tool.format_for_display() for tool in self.tools])
            
            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )
            
            self.messages = [{"role": "system", "content": system_message}]
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chat session: {e}")
            return False
    
    async def process_user_message(self, user_input: str) -> str:
        """
        Process a user message and get a response.
        
        Args:
            user_input: User's message
            
        Returns:
            Assistant's response
        """
        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})
            
            # Get LLM response
            llm_response = await self.llm_client.get_completion(self.messages)
            
            # Try to parse as a tool call
            tool_result = await self._try_execute_tool(llm_response)
            
            # Handle tool response if any
            if tool_result:
                # Add original LLM response to history
                self.messages.append({"role": "assistant", "content": llm_response})
                
                # Add tool result as system message
                tool_result_str = f"Tool execution result: {json.dumps(tool_result)}"
                self.messages.append({"role": "system", "content": tool_result_str})
                
                # Get final response from LLM
                final_response = await self.llm_client.get_completion(self.messages)
                self.messages.append({"role": "assistant", "content": final_response})
                return final_response
            else:
                # Just a regular response
                self.messages.append({"role": "assistant", "content": llm_response})
                return llm_response
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error: {error_msg}. Please try again."
    
    async def _try_execute_tool(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse the LLM response as a tool call and execute it.
        
        Args:
            llm_response: LLM response text
            
        Returns:
            Tool execution result or None if not a tool call
        """
        try:
            # Try to parse as JSON
            tool_call = json.loads(llm_response)
            
            # Check if it's a properly formatted tool call
            if "tool" in tool_call and "arguments" in tool_call:
                tool_name = tool_call["tool"]
                tool_args = tool_call["arguments"]
                
                logger.info(f"Executing tool: {tool_name}")
                logger.info(f"With arguments: {json.dumps(tool_args)}")
                
                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    return {"error": f"Tool '{tool_name}' not found"}
                
                # Validate arguments
                is_valid, error = tool.validate_arguments(tool_args)
                if not is_valid:
                    return {"error": f"Invalid arguments: {error}"}
                
                # Execute the tool
                return await self.server.execute_tool(tool_name, tool_args)
            
            return None
            
        except json.JSONDecodeError:
            # Not a JSON response, so not a tool call
            return None
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.server.cleanup()
        await self.llm_client.close()


async def interact_with_chat(session: ChatSession) -> None:
    """
    Interactive console for chat session.
    
    Args:
        session: Initialized chat session
    """
    print("\n=== MCP Client Chat ===")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'tools' to see available tools.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
                
            if user_input.lower() == "tools":
                print("\nAvailable tools:")
                for tool in session.tools:
                    print("\n" + tool.format_for_display())
                continue
                
            # Process user input
            print("\nThinking...")
            response = await session.process_user_message(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def main() -> None:
    """Main entry point."""
    config = Configuration()
    
    # Use the MCP server URL from the configuration
    server_url = config.mcp_server_url
    
    # Allow command-line override
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        logger.info(f"Using MCP server URL from command line: {server_url}")
    
    # Initialize components
    server_connection = ServerConnection(server_url)
    
    try:
        # Set up Azure OpenAI client with managed identity
        try:
            azure_credential = config.azure_credential
            llm_client = AzureOpenAIClient(
                credential=azure_credential,
                endpoint=config.azure_openai_endpoint,
                deployment=config.azure_openai_deployment,
                api_key=config.azure_openai_api_key  # Fallback if managed identity fails
            )
            print(f"Connected to Azure OpenAI at {config.azure_openai_endpoint}")
        except ValueError as e:
            print(f"Error: {e}")
            print("Azure OpenAI integration disabled. Only direct tool calling will be available.")
            llm_client = None
        
        # Initialize chat session
        chat_session = ChatSession(server_connection, llm_client)
        
        if not await chat_session.initialize():
            print("Failed to initialize chat session. Exiting.")
            return
        
        # Start interactive chat
        if llm_client:
            await interact_with_chat(chat_session)
        else:
            print("Azure OpenAI client not available. Please check your managed identity configuration.")
    
    finally:
        # Clean up resources
        await server_connection.cleanup()


if __name__ == "__main__":
    asyncio.run(main())