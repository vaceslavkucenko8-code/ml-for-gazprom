"""FastAPI application sharing the existing query/review implementation."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from .local_app import Application,PROJECT
from .storage import database_health


def create_app(root=None):
    application=Application(root or os.environ.get('RUNS_DIR'))
    @asynccontextmanager
    async def lifespan(app):
        yield
        application.pool.shutdown(wait=False,cancel_futures=True)
    app=FastAPI(title='Технологические сигналы',version='0.3',lifespan=lifespan)
    app.state.application=application

    @app.exception_handler(HTTPException)
    async def http_error(request,exc):
        return JSONResponse({'error':exc.detail},status_code=exc.status_code)

    @app.get('/',response_class=HTMLResponse)
    def index():
        return (PROJECT/'src/local_ui.html').read_text(encoding='utf-8').replace('__TOKEN__',application.token)

    @app.get('/review',response_class=HTMLResponse)
    def review_ui():
        return (PROJECT/'src/review_ui.html').read_text(encoding='utf-8').replace('__TOKEN__',application.token)

    @app.get('/submission',response_class=HTMLResponse)
    def submission():
        return (PROJECT/'src/submission.html').read_text(encoding='utf-8')

    @app.get('/health')
    def health():
        value=database_health()
        return JSONResponse(value,status_code=200 if value['ok'] else 503)

    @app.get('/api/runs/{run_id}/view')
    def presentation(run_id: str, translate_cards: bool=True, candidate_id: str | None=None):
        from .presentation_view import load_view
        try: return load_view(application.folder(run_id), translate_cards=translate_cards, candidate_id=candidate_id)
        except FileNotFoundError: raise HTTPException(404, 'Результат ещё не готов или отсутствует')
        except ValueError: raise HTTPException(400, 'Некорректные данные запуска')

    @app.get('/api/runs')
    def runs():return application.list_runs()

    async def body(request):
        if request.headers.get('X-Local-Token')!=application.token:raise HTTPException(403,'Обновите страницу приложения')
        raw=await request.body()
        if len(raw)>1_000_000:raise HTTPException(413,'Слишком большой запрос')
        try:
            value=await request.json()
            if not isinstance(value,dict):raise ValueError()
            return value
        except Exception:raise HTTPException(400,'Некорректный JSON')

    @app.post('/api/search',status_code=202)
    async def search(request:Request):
        data=await body(request)
        try:return {'run_id':application.submit(data)}
        except (ValueError,TypeError,KeyError) as exc:raise HTTPException(400,str(exc))

    @app.post('/api/review')
    async def review(request:Request):
        data=await body(request)
        try:return await asyncio.to_thread(lambda:{'run_id':application.review(data)})
        except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:raise HTTPException(400,str(exc))

    @app.get('/runs/{run_id}/{filename}')
    def artifact(run_id:str,filename:str):
        if filename not in ('report.html','result.json','automatic.json','detector_input.json','reviews.json','summary.json'):
            raise HTTPException(404,'Не найдено')
        try:path=application.folder(run_id)/filename
        except ValueError:raise HTTPException(400,'Некорректный ID')
        if not path.is_file():raise HTTPException(404,'Не найдено')
        return FileResponse(path,media_type='text/html' if filename.endswith('.html') else 'application/json')
    return app


app=create_app()
