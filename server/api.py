from fastapi import FastAPI
import uvicorn
from server.modules.inference import load_model_at_run
from server.modules.docs import description
from server.api.demo import demo_router
from server.api.fots import fots_router
from server.api.portfolio import portfolio_router

app = FastAPI(description=description)
app.include_router(fots_router)
app.include_router(demo_router)
app.include_router(portfolio_router)


#메인페이지 실행할때 모델로드
@app.on_event("startup")
def startup_event():
    print("application Start!!!")
    load_model_at_run()

@app.on_event("shutdown")
def startdown_event():
    print("Application Stop.")

@app.get("/")
def read_root():
    return "Welcome Boost Camp AI tech CV7's API"

if __name__ == '__main__':
    uvicorn.run('api:app', port=6006, host='0.0.0.0', reload=True)