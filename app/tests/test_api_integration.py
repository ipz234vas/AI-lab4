"""Local HTTP integration tests. Mock only the Ollama side, not the app server."""
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock,patch, MagicMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server

def completion(content, prompt_tokens=None, completion_tokens=None):
    usage = (SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
             if prompt_tokens is not None or completion_tokens is not None else None)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.port=cls.httpd.server_address[1]
        cls.t=threading.Thread(target=cls.httpd.serve_forever,daemon=True);cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown();cls.httpd.server_close();cls.t.join(3)

    def setUp(self):
        server._connections.clear()

    def req(self,path,body=None):
        url=f'http://127.0.0.1:{self.port}{path}'
        data=json.dumps(body).encode() if body is not None else None
        r=Request(url,data=data,headers={'Content-Type':'application/json'})
        try:
            with urlopen(r,timeout=4) as res:
                return res.status,json.load(res)
        except HTTPError as exc:
            return exc.code,json.load(exc)

    @patch('server.ollama_api')
    def test_models_status(self,api):
        api.return_value={'models':[{'name':server.BASE_MODEL},{'name':'qwen3:4b'}]}
        code,result=self.req('/api/models')
        self.assertEqual(code,200)
        self.assertEqual(result['models'],['qwen3.5:2b-q4_K_M','qwen3:4b'])
        self.assertEqual(result['connected'],[])

    @patch('server.make_client')
    @patch('server.ollama_api')
    def test_connect_classify_change_temp_and_reply(self,api,make):
        rawmodel='qwen3:4b'
        api.side_effect=[{'models':[{'name':rawmodel}]},{'status':'success'}]
        warm=MagicMock()
        warm.__enter__.return_value=warm
        warm.chat.completions.create.return_value=completion('READY')
        make.return_value=warm
        code,connected=self.req('/api/connect',{'model':rawmodel,'num_ctx':2048})
        self.assertEqual(code,200)
        self.assertTrue(connected['connected'])
        self.assertIn('warmup',connected)
        with patch('server.classify') as classify, patch('server.generate_reply') as reply:
            classify.return_value=({'urgency_level':'medium','reason':'До вечора.'},{'elapsed_s':1.2})
            reply.return_value=({'suggested_reply':'Добре, уточню.'},{'elapsed_s':0.9})
            payload={'model':rawmodel,'num_ctx':2048,'max_tokens':128,'sender':'керівник','message':'Зроби до вечора'}
            code,r=self.req('/api/classify',{**payload,'temperature':0.1,'seed':1})
            self.assertEqual(code,200);self.assertEqual(r['result']['urgency_level'],'medium')
            self.assertEqual(r['settings']['temperature'],0.1)
            self.assertEqual(r['settings']['seed'],1)
            self.assertEqual(classify.call_args.args[1].seed,1)
            code,r=self.req('/api/reply',{**payload,'temperature':0.7,'urgency_level':'medium','seed':0})
            self.assertEqual(code,200);self.assertEqual(r['result']['suggested_reply'],'Добре, уточню.')
            self.assertEqual(reply.call_args.args[1].temperature,0.7)
            self.assertEqual(reply.call_args.args[1].seed,0)
            self.assertEqual(r['settings']['seed'],0)
            self.assertEqual(classify.call_count,1)  # reply does NOT re-classify
            code,r=self.req('/api/classify',{**payload,'num_ctx':8192,'temperature':0})
            self.assertEqual(code,409)   # must connect a new context before using it
            self.assertIn('Підключити',r['error'])

if __name__=='__main__': unittest.main()
