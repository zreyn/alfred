from fastmcp import FastMCP

mcp = FastMCP("mcp-service")


@mcp.tool()
def hello(name: str = "World") -> str:
    """Say hello to someone.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting message.
    """
    return f"Hello, {name}!"


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a greeting resource for a specific name."""
    return f"Hello, {name}! Welcome to the MCP service."


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
