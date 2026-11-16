"""Loopback-only, manifest-whitelisted read-only viewer server with media seeking."""
import argparse,hashlib,json,os,re
from pathlib import Path
from datetime import datetime,timezone
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,unquote

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--receipt",required=True);p.add_argument("--expected-files",type=int,default=10);a=p.parse_args()
    root=Path(a.root).resolve();source=json.loads(Path(a.manifest).read_text(encoding="utf8"));files=source["payload"]["files"]
    allowed={r["name"]:r for r in files};assert len(allowed)==len(files)==a.expected_files and a.expected_files>0
    for name,row in allowed.items():
        f=(root/name).resolve();assert f.is_relative_to(root) and f.is_file()
        assert hashlib.sha256(f.read_bytes()).hexdigest()==row["sha256"]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def serve(self,body):
            name=unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
            if name not in allowed:self.send_error(404);return
            target=(root/name).resolve()
            if not target.is_relative_to(root):self.send_error(404);return
            size=target.stat().st_size;start=0;end=size-1;partial=False
            header=self.headers.get("Range")
            if header:
                match=re.fullmatch(r"bytes=(\d*)-(\d*)",header)
                if not match or not any(match.groups()):self.send_error(416);return
                x,y=match.groups()
                if x:start=int(x);end=min(int(y),end) if y else end
                else:start=max(0,size-int(y))
                if start<0 or start>end or start>=size:self.send_error(416);return
                partial=True
            self.send_response(206 if partial else 200)
            content={".html":"text/html; charset=utf-8",".js":"application/javascript; charset=utf-8",".css":"text/css; charset=utf-8",".json":"application/json; charset=utf-8",".wav":"audio/wav"}[target.suffix]
            self.send_header("Content-Type",content);self.send_header("Content-Length",str(end-start+1));self.send_header("Accept-Ranges","bytes")
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
            if partial:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            if body:
                try:
                    with target.open("rb") as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            data=f.read(min(65536,remaining))
                            if not data:break
                            self.wfile.write(data);remaining-=len(data)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def do_GET(self):self.serve(True)
        def do_HEAD(self):self.serve(False)
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler);server.daemon_threads=True
    receipt=Path(a.receipt);assert not receipt.exists()
    receipt.write_text(json.dumps(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),host="127.0.0.1",port=server.server_port,root=str(root),
        allowed_files=sorted(allowed),read_only=True,directory_listing=False,public_interface=False),indent=2)+"\n",encoding="utf8")
    print("R34_READONLY_VIEWER",server.server_port,flush=True)
    try:server.serve_forever()
    finally:server.server_close()
if __name__=="__main__":main()
