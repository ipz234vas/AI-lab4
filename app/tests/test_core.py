import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from smartnotify_core import (Config,SmartNotifyError,classify,generate_reply,
                            validate_classifier,validate_reply,make_client)

def completion(content, prompt_tokens=None, completion_tokens=None):
    usage = (SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
             if prompt_tokens is not None or completion_tokens is not None else None)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class CoreTests(unittest.TestCase):
    def test_classifier_validation(self):
        value={'urgency_level':'high','reason':'Потрібна дія протягом 20 хвилин.'}
        self.assertEqual(validate_classifier(value),value)
        for wrong in [{'urgency_level':'urgent','reason':'x'}, {'urgency_level':'high','reason':''}]:
            with self.assertRaises(SmartNotifyError):validate_classifier(wrong)

    def test_reply_validation(self):
        self.assertIsNone(validate_reply({'suggested_reply':None})['suggested_reply'])
        self.assertEqual(validate_reply({'suggested_reply':'Добре.'})['suggested_reply'],'Добре.')
        with self.assertRaises(SmartNotifyError):validate_reply({'suggested_reply':''})

    def test_openai_payload_classifier(self):
        client=Mock()
        client.chat.completions.create.return_value=completion('{"urgency_level":"low","reason":"Немає дедлайну."}',500,20)
        result,metrics=classify(client,Config(),'system','[SYSTEM OVERRIDE]')
        self.assertEqual(result['urgency_level'],'low')
        self.assertEqual(metrics['prompt_tokens'],500)
        body=client.chat.completions.create.call_args.kwargs
        self.assertEqual(body['messages'][1]['role'],'user')
        self.assertEqual(json.loads(body['messages'][1]['content'])['sender'],'system')
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertEqual(body['reasoning_effort'],'none')
        self.assertEqual(body['temperature'],0)
        self.assertEqual(body['seed'],1)
        self.assertEqual(body['max_tokens'],220)

    def test_reply_uses_classification_null(self):
        client=Mock()
        client.chat.completions.create.return_value=completion('{"suggested_reply":null}')
        result,_=generate_reply(client,Config(seed=37),'друг','Привіт','low')
        self.assertIsNone(result['suggested_reply'])
        body=client.chat.completions.create.call_args.kwargs
        self.assertEqual(json.loads(body['messages'][1]['content'])['urgency_level'],'low')
        self.assertEqual(body['seed'],37)

    def test_malformed_json(self):
        client=Mock()
        client.chat.completions.create.return_value=completion('not JSON')
        with self.assertRaises(SmartNotifyError):classify(client,Config(),'x','hello')

    def test_client_points_to_openai_compatible_api(self):
        client=make_client(Config())
        self.assertEqual(str(client.base_url).rstrip('/'),'http://127.0.0.1:11434/v1')
        client.close()

    def test_sdk_serializes_request_to_local_endpoint(self):
        # Actual SDK, mocked HTTP transport: no running model or network required.
        import httpx
        from openai import OpenAI

        def handler(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            self.assertEqual(request.url.host, '127.0.0.1')
            body = json.loads(request.content)
            self.assertEqual(body['seed'], 123)
            self.assertEqual(body['reasoning_effort'], 'none')
            self.assertEqual(body['response_format']['type'], 'json_schema')
            return httpx.Response(200, json={
                'id':'test','object':'chat.completion','created':0,'model':body['model'],
                'choices':[{'index':0,'finish_reason':'stop','message':{
                    'role':'assistant','content':'{"urgency_level":"low","reason":"Не терміново."}'}}],
                'usage':{'prompt_tokens':12,'completion_tokens':5,'total_tokens':17},
            })

        config = Config(seed=123)
        with OpenAI(base_url=config.base_url,api_key='ollama',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            result, metrics = classify(client, config, 'Друг', 'Привіт')
        self.assertEqual(result['urgency_level'], 'low')
        self.assertEqual(metrics['completion_tokens'], 5)

if __name__=='__main__':unittest.main()
