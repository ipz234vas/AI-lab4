import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from server import (AppError, BASE_MODEL, CONTEXTS, alias_for, connect,
                    context_model, installed_models, parse_selection, read_settings)

class ServerTests(unittest.TestCase):
    def setUp(self):
        with server._lock:
            server._connections.clear()

    def tearDown(self):
        with server._lock:
            server._connections.clear()

    def test_selection_validation(self):
        self.assertEqual(parse_selection({'model':BASE_MODEL}), (BASE_MODEL, 4096))
        for data in ({}, {'model':42}, {'model':' '}, {'model':BASE_MODEL,'num_ctx':100},
                     {'model':BASE_MODEL,'num_ctx':'invalid'}):
            with self.subTest(data=data), self.assertRaises(AppError):
                parse_selection(data)

    def test_dynamic_model_aliases(self):
        self.assertNotEqual(alias_for('qwen3.5:2b-q4_K_M', 4096), alias_for('qwen3:4b-q4_K_M', 4096))
        self.assertNotEqual(alias_for('qwen3.5:2b-q4_K_M', 4096), alias_for('qwen3.5:2b-q4_K_M', 8192))
        self.assertEqual(alias_for(BASE_MODEL, 2048),alias_for(BASE_MODEL, 2048))
        self.assertEqual(CONTEXTS,(1024,1536,2048,3072,4096,8192))

    @patch('server.ollama_api')
    def test_installed_models_no_app_aliases(self, api):
        api.return_value={'models':[{'name': 'qwen3:4b'}, {'name':alias_for(BASE_MODEL,4096)}, {'name':BASE_MODEL}]}
        self.assertEqual(installed_models(),sorted([BASE_MODEL,'qwen3:4b'],key=str.casefold))

    @patch('server.ollama_api')
    def test_context_alias_creation_existing_and_uninstalled_rejected(self, api):
        model='qwen3:4b-q4_K_M'
        api.side_effect=[{'models':[{'name':model}]},{'status':'success'}]
        alias=context_model(model,2048)
        self.assertEqual(alias,alias_for(model,2048))
        self.assertEqual(api.call_args_list[1].args[0],'/api/create')
        payload=api.call_args_list[1].args[1]
        self.assertEqual(payload['from'],model)
        self.assertEqual(payload['parameters'],{'num_ctx':2048})
        self.assertFalse(payload['stream'])
        api.reset_mock()
        api.return_value={'models':[{'name':model},{'name':alias}]};api.side_effect=None
        self.assertEqual(context_model(model,2048),alias)
        self.assertEqual(api.call_count,1)
        api.return_value={'models':[{'name':'other:model'}]}
        with self.assertRaises(AppError):context_model(model,4096)

    def test_settings_require_connection_and_persist(self):
        with self.assertRaises(AppError) as cm:
            read_settings({'model':BASE_MODEL})
        self.assertEqual(cm.exception.status,409)
        with server._lock:
            server._connections[(BASE_MODEL,4096)]=alias_for(BASE_MODEL,4096)
        config, values=read_settings({'model':BASE_MODEL,'temperature':0.4,'num_ctx':4096,'max_tokens':128})
        self.assertEqual(config.model,alias_for(BASE_MODEL,4096))
        self.assertAlmostEqual(config.temperature,.4)
        self.assertEqual(config.max_tokens,128)
        self.assertEqual(values['model'],BASE_MODEL)
        with self.assertRaises(AppError) as cm:
            read_settings({'model':BASE_MODEL,'num_ctx':8192})
        self.assertEqual(cm.exception.status,409)
        for data in ({'temperature':float('nan')},{'temperature':1.5},{'temperature':'oops'},
                     {'max_tokens':999}):
            with self.subTest(data=data), self.assertRaises(AppError):
                read_settings({'model':BASE_MODEL,**data})

    @patch('server.make_client')
    @patch('server.context_model')
    def test_connect_warms_via_openai_and_registers_after_success(self, ctx, make):
        ctx.return_value='warm:ctx4096'
        client=Mock();make.return_value=client
        client.create.return_value={'choices':[{'message':{'content':'READY'}}],
                                    'usage':{'prompt_tokens':11,'completion_tokens':1}}
        info=connect({'model':'qwen3:4b','num_ctx':4096})
        self.assertTrue(info['connected'])
        self.assertEqual(info['model'],'qwen3:4b')
        self.assertEqual(info['warmup']['completion_tokens'],1)
        self.assertIn(('qwen3:4b',4096),server._connections)
        body=client.create.call_args.args[0]
        self.assertEqual(body['model'],'warm:ctx4096')
        self.assertEqual(body['reasoning_effort'],'none')
        self.assertEqual(body['max_tokens'],12)
        self.assertEqual(make.call_args.args[0].base_url,'http://127.0.0.1:11434/v1')

    @patch('server.make_client')
    @patch('server.context_model',return_value='test:ctx2048')
    def test_failed_warmup_does_not_register(self, ctx, make):
        make.return_value.create.return_value={'choices':[{'message':{'content':''}}]}
        with self.assertRaises(AppError):connect({'model':BASE_MODEL,'num_ctx':2048})
        self.assertNotIn((BASE_MODEL,2048),server._connections)

if __name__=='__main__': unittest.main()
