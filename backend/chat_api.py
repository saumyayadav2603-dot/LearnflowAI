"""HTTP endpoint for AI chat. Defaults to 503 until securely configured."""
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, HTTPException, Request

from .contracts import ChatRequest, ChatResponse
from .chat_service import (
    AuthorizationError,
    ChatUnavailableError,
    GroundedChatService,
)

router = APIRouter(tags=["chat"])
DEMO_NETWORKS = tuple(
    ip_network(network)
    for network in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fc00::/7",
    )
)


def is_demo_network_peer(request: Request) -> bool:
    peer = request.client
    if peer is None:
        return False
    try:
        address = ip_address(peer.host)
    except ValueError:
        return False
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return any(address in network for network in DEMO_NETWORKS)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    if not is_demo_network_peer(request):
        raise HTTPException(
            status_code=403,
            detail="Chat is restricted to localhost or private network clients.",
        )
    service: GroundedChatService | None = getattr(
        request.app.state, "chat_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=503, detail="Chat integrations are not configured."
        )
    try:
        return await service.answer(payload)
    except AuthorizationError as exc:
        raise HTTPException(
            status_code=403, detail="Requested sources are not permitted."
        ) from exc
    except ChatUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
