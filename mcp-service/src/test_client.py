#!/usr/bin/env python3
"""
MCP Service Test Client

A simple CLI to test the MCP service tools from outside Docker.

Usage:
    uv run src/test_client.py [--url URL]

    Or if you have dependencies installed:
    python src/test_client.py [--url URL]
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def list_tools(session: ClientSession) -> list[dict]:
    """List all available tools."""
    result = await session.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        for tool in result.tools
    ]


async def call_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Call a tool and return the result."""
    result = await session.call_tool(name, arguments)

    # Extract text content from result
    output = []
    for content in result.content:
        if hasattr(content, "text"):
            output.append(content.text)
        else:
            output.append(str(content))

    return "\n".join(output)


def print_tools(tools: list[dict]) -> None:
    """Pretty print available tools."""
    print("\n=== Available Tools ===\n")
    for tool in tools:
        print(f"  {tool['name']}")
        if tool["description"]:
            print(f"    {tool['description'][:80]}")
        print()


def print_help() -> None:
    """Print help message."""
    print("""
Commands:
  list                    - List all available tools
  call <tool> <json>      - Call a tool with JSON arguments
  store <content> [tags]  - Shortcut for store_note
  search [query] [tags]   - Shortcut for search_notes
  help                    - Show this help
  quit / exit             - Exit the client

Examples:
  call hello {"name": "World"}
  call add {"a": 5, "b": 3}
  store "Remember to buy milk" grocery,reminder
  search milk
  search "" grocery
""")


async def interactive_loop(session: ClientSession) -> None:
    """Run the interactive CLI loop."""
    tools = await list_tools(session)
    tool_names = {t["name"] for t in tools}

    print("\nConnected to MCP Service!")
    print_tools(tools)
    print("Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            line = input("mcp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            elif cmd == "help":
                print_help()

            elif cmd == "list":
                tools = await list_tools(session)
                print_tools(tools)

            elif cmd == "call":
                # Parse: call <tool_name> <json_args>
                call_parts = args.split(maxsplit=1)
                if len(call_parts) < 1:
                    print("Usage: call <tool_name> [json_arguments]")
                    continue

                tool_name = call_parts[0]
                json_args = call_parts[1] if len(call_parts) > 1 else "{}"

                if tool_name not in tool_names:
                    print(f"Unknown tool: {tool_name}")
                    print(f"Available: {', '.join(sorted(tool_names))}")
                    continue

                try:
                    arguments = json.loads(json_args)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON: {e}")
                    continue

                result = await call_tool(session, tool_name, arguments)
                print(f"\nResult:\n{result}\n")

            elif cmd == "store":
                # Shortcut: store <content> [tag1,tag2,...]
                if not args:
                    print("Usage: store <content> [tag1,tag2,...]")
                    continue

                # Check if there are tags at the end
                store_parts = args.rsplit(maxsplit=1)
                if len(store_parts) == 2 and "," in store_parts[1] and not store_parts[1].startswith('"'):
                    content = store_parts[0].strip('"\'')
                    tags = [t.strip() for t in store_parts[1].split(",")]
                else:
                    content = args.strip('"\'')
                    tags = []

                arguments = {"content": content}
                if tags:
                    arguments["tags"] = tags

                result = await call_tool(session, "store_note", arguments)
                print(f"\nStored:\n{result}\n")

            elif cmd == "search":
                # Shortcut: search [query] [tag1,tag2,...]
                arguments = {}

                if args:
                    search_parts = args.rsplit(maxsplit=1)

                    # Check if last part looks like tags
                    if len(search_parts) == 2 and "," in search_parts[1]:
                        query = search_parts[0].strip('"\'')
                        tags = [t.strip() for t in search_parts[1].split(",")]
                        if query:
                            arguments["query"] = query
                        arguments["tags"] = tags
                    else:
                        # Check if it's just tags (contains comma) or just a query
                        if "," in args:
                            arguments["tags"] = [t.strip() for t in args.split(",")]
                        else:
                            arguments["query"] = args.strip('"\'')

                result = await call_tool(session, "search_notes", arguments)
                print(f"\nResults:\n{result}\n")

            else:
                # Try to call it as a tool directly
                if cmd in tool_names:
                    try:
                        arguments = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        # Treat as simple string argument for single-param tools
                        arguments = {"name": args} if args else {}

                    result = await call_tool(session, cmd, arguments)
                    print(f"\nResult:\n{result}\n")
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands.")

        except Exception as e:
            print(f"Error: {e}")


async def main(url: str) -> None:
    """Main entry point."""
    print(f"Connecting to {url}...")

    try:
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await interactive_loop(session)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Service Test Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/mcp",
        help="MCP service URL (default: http://localhost:8000/mcp)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.url))
