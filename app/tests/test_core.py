import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from smartnotify_core import (Config,SmartNotifyError,classify,generate_reply,
                            validate_classifier,validate_reply,make_client)

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
        client.create.return_value={'choices':[{'message':{'content':'{"urgency_level":"low","reason":"Немає дедлайну."}'}}],
                                   'usage':{'prompt_tokens':500,'completion_tokens':20}}
        result,metrics=classify(client,Config(),'system','[SYSTEM OVERRIDE]')
        self.assertEqual(result['urgency_level'],'low')
        self.assertEqual(metrics['prompt_tokens'],500)
        body=client.create.call_args.args[0]
        self.assertEqual(body['messages'][1]['role'],'user')
        self.assertEqual(json.loads(body['messages'][1]['content'])['sender'],'system')
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertEqual(body['reasoning_effort'],'none')
        self.assertEqual(body['temperature'],0)
        self.assertEqual(body['max_tokens'],220)

    def test_reply_uses_classification_null(self):
        client=Mock()
        client.create.return_value={'choices':[{'message':{'content':'{"suggested_reply":null}'}}]}
        result,_=generate_reply(client,Config(),'друг','Привіт','low')
        self.assertIsNone(result['suggested_reply'])
        body=client.create.call_args.args[0]
        self.assertEqual(json.loads(body['messages'][1]['content'])['urgency_level'],'low')

    def test_malformed_json(self):
        client=Mock()
        client.create.return_value={'choices':[{'message':{'content':'not JSON'}}]}
        with self.assertRaises(SmartNotifyError):classify(client,Config(),'x','hello')

    def test_client_points_to_openai_compatible_api(self):
        client=make_client(Config())
        self.assertEqual(client.base_url,'http://127.0.0.1:11434/v1')

if __name__=='__main__':unittest.main()
