from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from server.api.ParkJinhyung_Portfolio import pofol_html 
portfolio_router = APIRouter(prefix='/portfolio')

@portfolio_router.get("/", response_class=HTMLResponse)
async def read_portfolio(request: Request):
    return pofol_html