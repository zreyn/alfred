from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

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


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
