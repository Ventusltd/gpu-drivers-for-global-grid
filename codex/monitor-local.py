"""Read-only loopback progress display for existing overnight controllers."""
import argparse
import datetime as dt
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import psutil

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--schedule',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--learning',type=Path)
parser.add_argument('--port',type=int,default=8978)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
def read(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError): return {}
run=read(args.run/'RUN.json')
deadline=dt.datetime.fromisoformat(run['startedAt']).timestamp()+run['hours']*3600
def state():
    s=read(args.run/'STATUS.json'); plan=read(args.schedule/'PROGRESS.json')
    age=time.time()-dt.datetime.fromisoformat(s['at']).timestamp() if s.get('at') else None
    try:
        process=psutil.Process(run['pid'])
        alive=any('overnight.py' in a for a in process.cmdline())
    except psutil.Error: alive=False
    learning=read(args.learning/'STATUS.json') if args.learning else {}
    learningFinal=read(args.learning/'FINAL.json') if args.learning else {}
    return {'checkedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'supervisorAlive':alive,'heartbeatAgeSeconds':round(age,1) if age is not None else None,'alert': 'Supervisor absent or heartbeat stale' if not alive or age is None or age>60 else None,'status':s,'learning':learning,'learningFinal':learningFinal,'verified':sum(v=='verified' for v in plan.get('states',{}).values()),'scopeCount':len(plan.get('states',{})),'planAt':plan.get('at'),'deadlineUTC':dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat()}
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/status.json': payload=json.dumps(state()).encode();kind='application/json'
        elif self.path=='/': payload=Path(__file__).with_name('monitor-local.html').read_bytes();kind='text/html; charset=utf-8'
        else: self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
    def log_message(self,*args): pass
server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
def watch():
    while time.time()<deadline+60 and not (args.out/'STOP').exists():
        value=state();(args.out/'latest.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        # Compact summaries only; bounded to approximately 960 records over eight hours.
        with (args.out/'health.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({k:value[k] for k in ['checkedAt','supervisorAlive','heartbeatAgeSeconds','alert','verified']})+'\n')
        time.sleep(30)
    server.shutdown()
threading.Thread(target=watch,daemon=True).start()
try: server.serve_forever()
finally: server.server_close()
