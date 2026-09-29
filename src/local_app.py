"""Local-only search and evidence review UI. File-backed handoff MVP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import webbrowser

from .pipeline import process_export

PROJECT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, root=None):
        self.root = Path(root or PROJECT/'runs')
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job_dir = self.root/'.jobs'
        self.job_dir.mkdir(exist_ok=True)
        self._load_jobs()

    def _load_jobs(self):
        for path in self.job_dir.glob('*.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                self.folder(job['run_id'])
                # A second reader (tests, another local UI) does not own the
                # worker and must not declare its live job interrupted.
                deadline = job.get('deadline_at', path.stat().st_mtime + 660)
                if job['status'] in ('queued','running') and time.time() > deadline:
                    job.update(status='failed', error='Поиск превысил контрольный срок либо сервер был остановлен. Запустите поиск повторно.')
                    self.jobs[job['run_id']] = job
                    self._save_job(job['run_id'])
                self.jobs[job['run_id']] = job
            except (ValueError, KeyError, OSError):
                continue

    def _save_job(self, rid):
        path = self.job_dir/(rid+'.json')
        temp = path.with_suffix('.' + secrets.token_hex(6) + '.tmp')
        temp.write_text(json.dumps(self.jobs[rid],ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(path)

    def folder(self, run_id):
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
            raise ValueError('Некорректный ID запуска')
        return self.root/run_id

    def list_runs(self):
        results = []
        for folder in sorted(self.root.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True):
            if (folder/'summary.json').is_file():
                status='done' if not (folder/'result.json').exists() or (folder/'completion.json').exists() else 'failed'
                results.append({'run_id': folder.name, 'status': status,
                    'error':None if status=='done' else 'Запуск не завершён; проверьте сохранение результатов и журнал сервера.',
                    'summary': json.loads((folder/'summary.json').read_text(encoding='utf-8'))})
        with self.lock:
            self._load_jobs()
            known = {r['run_id'] for r in results}
            results.extend(dict(j) for rid, j in self.jobs.items() if rid not in known)
        return results

    def submit(self, data):
        query = data.get('query', '').strip()
        if not 3 <= len(query) <= 300:
            raise ValueError('Запрос должен содержать от 3 до 300 символов')
        search_query = data.get('search_query', '').strip()
        if len(search_query)>300:
            raise ValueError('Слишком длинный поисковый запрос')
        as_of = data.get('as_of') or datetime.now(timezone.utc).date().isoformat()
        datetime.strptime(as_of, '%Y-%m-%d')
        mode = data.get('search_mode', 'extended')
        if mode not in ('quick', 'extended'):
            raise ValueError('Неизвестный режим поиска')
        rid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(3)
        with self.lock:
            self._load_jobs()
            if any(j['status'] in ('queued','running') for j in self.jobs.values()):
                raise ValueError('Дождитесь завершения текущего поиска')
            self.jobs[rid] = {'run_id':rid, 'status':'queued', 'query':query, 'search_mode':mode,
                              'deadline_at': time.time() + (660 if mode == 'extended' else 420)}
            self._save_job(rid)
        self.pool.submit(self._search, rid, query, search_query, as_of, mode)
        return rid

    def _search(self, rid, query, search_query, as_of, mode='extended'):
        with self.lock:
            self.jobs[rid]['status'] = 'running'
            self._save_job(rid)
        extended = mode == 'extended'
        command = [sys.executable, '-m', 'src.pipeline', '--query', query, '--as-of', as_of,
                   '--out', str(self.folder(rid)), '--results-per-connector', '20' if extended else '4',
                   '--max-queries', '3' if extended else '1',
                   '--search-budget-seconds', '360' if extended else '240']
        if extended:
            command.append('--enrich-sources')
        if search_query:
            command += ['--search-query', search_query]
        try:
            import os
            run = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True,
                                 encoding='utf-8', timeout=600 if extended else 360, env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            if run.returncode:
                raise ValueError(run.stderr[-2000:] or 'Ошибка запуска')
            with self.lock:
                self.jobs[rid]['status'] = 'done'
                self._save_job(rid)
        except Exception as exc:
            with self.lock:
                self.jobs[rid].update(status='failed', error=str(exc))
                self._save_job(rid)

    def review(self, data):
        source = self.folder(data['run_id'])
        if not isinstance(data.get('reviews'), list) or not data['reviews']:
            raise ValueError('Нет проверенных утверждений')
        payload = json.loads((source/'retrieved.json').read_text(encoding='utf-8'))
        summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
        existing = json.loads((source/'reviews.json').read_text(encoding='utf-8'))
        merged = {(r['candidate_id'],r['evidence_id'],r['feature']):r for r in existing}
        for r in data['reviews']:
            if r.get('reviewed') is not True or r.get('scope_matches_candidate') is not True:
                raise ValueError('Подтвердите смысл цитаты и её применимость')
            merged[(r['candidate_id'],r['evidence_id'],r['feature'])] = r
        rid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-review-' + secrets.token_hex(3)
        process_export(payload, summary['as_of'], self.folder(rid), list(merged.values()),
                       summary.get('max_age_days',540), payload.get('search_query'))
        return rid


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json; charset=utf-8'):
            raw = json.dumps(body,ensure_ascii=False).encode('utf-8') if isinstance(body,(dict,list)) else body
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                if self.path == '/':
                    page=(PROJECT/'src/local_ui.html').read_text(encoding='utf-8').replace('__TOKEN__',app.token)
                    return self.send(200,page.encode('utf-8'),'text/html; charset=utf-8')
                if self.path == '/api/runs':
                    return self.send(200,app.list_runs())
                parts=self.path.strip('/').split('/')
                if len(parts)==3 and parts[0]=='runs' and parts[2] in ('report.html','detector_input.json','reviews.json','summary.json'):
                    path=app.folder(parts[1])/parts[2]
                    return self.send(200,path.read_bytes(),'text/html; charset=utf-8' if path.suffix=='.html' else 'application/json; charset=utf-8')
                self.send(404,{'error':'Не найдено'})
            except (ValueError,FileNotFoundError) as exc:
                self.send(400,{'error':str(exc)})

        def do_POST(self):
            if self.headers.get('X-Local-Token') != app.token:
                return self.send(403,{'error':'Обновите страницу приложения'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1_000_000:
                    raise ValueError('Недопустимый размер запроса')
                data=json.loads(self.rfile.read(size))
                if self.path=='/api/search':
                    return self.send(202,{'run_id':app.submit(data)})
                if self.path=='/api/review':
                    return self.send(200,{'run_id':app.review(data)})
                self.send(404,{'error':'Не найдено'})
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
                self.send(400,{'error':str(exc)})

        def log_message(self,*args):
            pass
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    import uvicorn
    url=f'http://127.0.0.1:{args.port}'
    print(f'Open {url}; Ctrl+C to stop.',flush=True)
    if not args.no_browser:
        threading.Timer(1,lambda:webbrowser.open(url)).start()
    uvicorn.run('src.api:app',host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
