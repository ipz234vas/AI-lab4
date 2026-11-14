"""SmartNotify: local inference via Ollama OpenAI-compatible HTTP endpoint.

Only validates JSON schema. No semantic filtering, keyword rules, or templated replies.
Uses Python standard library; no OpenAI SDK required.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

PROMPTS = Path(__file__).resolve().parent / 'prompts'
LEVELS = ('low','medium','high','critical')

CLASSIFIER_SCHEMA = {
    'type':'object','properties':{
        'urgency_level':{'type':'string','enum':list(LEVELS)},
        'reason':{'type':'string'}},
    'required':['urgency_level','reason'],'additionalProperties':False,
}
REPLY_SCHEMA = {
    'type':'object','properties':{'suggested_reply':{'type':['string','null']}},
    'required':['suggested_reply'],'additionalProperties':False,
}

@dataclass(frozen=True)
class Config:
    model: str = 'qwen3.5:2b-q4_K_M'
    base_url: str = 'http://127.0.0.1:11434/v1'
    temperature: float = 0
    seed: int = 1
    max_tokens: int = 220
    target_num_ctx: int = 4096
    timeout_seconds: float = 120

class SmartNotifyError(Exception):
    pass

class OpenAICompatibleClient:
    def __init__(self, config:Config):
        self.base_url=config.base_url
        self.timeout=config.timeout_seconds

    def create(self, data:dict) -> dict:
        req = Request(self.base_url + '/chat/completions',
            data=json.dumps(data,ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type':'application/json'},method='POST')
        try:
            with urlopen(req,timeout=self.timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except HTTPError as exc:
            detail=exc.read(450).decode('utf-8','replace')
            if exc.code==404:
                raise SmartNotifyError(f'Модель не знайдена в Ollama: {detail}') from exc
            raise SmartNotifyError(f'Помилка Ollama HTTP {exc.code}: {detail}') from exc
        except TimeoutError as exc:
            raise SmartNotifyError(f'Таймаут Ollama ({self.timeout:.0f} с).') from exc
        except (URLError,OSError) as exc:
            raise SmartNotifyError('Не вдалося підключитися до Ollama. Перевір, чи запущено сервер.') from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SmartNotifyError('Ollama повернула некоректний JSON API.') from exc

def make_client(config:Config) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config)


def request_json(client:OpenAICompatibleClient, config:Config,
                 prompt_file:str, input_data:dict, schema:dict, schema_name:str):
    prompt=(PROMPTS/prompt_file).read_text(encoding='utf-8')
    body = {
        'model':config.model,
        'messages':[
            {'role':'system','content':prompt},
            {'role':'user','content':json.dumps(input_data,ensure_ascii=False)},
        ],
        'temperature':config.temperature,'seed':config.seed,
        'max_tokens':config.max_tokens,'reasoning_effort':'none',
        'stream':False,
        'response_format':{'type':'json_schema','json_schema':{
            'name':schema_name,'strict':True,'schema':schema}},
    }
    started=perf_counter()
    response=client.create(body)
    elapsed=perf_counter()-started
    try:
        content=response['choices'][0]['message']['content']
        if not isinstance(content,str) or not content.strip():
            raise ValueError('Порожня відповідь моделі')
        answer=json.loads(content)
        if not isinstance(answer,dict):raise ValueError('Очікується JSON-об’єкт')
    except (KeyError,IndexError,ValueError,TypeError,json.JSONDecodeError) as exc:
        raise SmartNotifyError(f'Модель повернула некоректний JSON: {exc}') from exc
    usage=response.get('usage') or {}
    return answer,{
        'elapsed_s':round(elapsed,3),
        'prompt_tokens':usage.get('prompt_tokens'),
        'completion_tokens':usage.get('completion_tokens'),
    }


def validate_classifier(value:dict)->dict:
    if set(value)!={'urgency_level','reason'} or value.get('urgency_level') not in LEVELS or not isinstance(value.get('reason'),str) or not value['reason'].strip():
        raise SmartNotifyError('Модель повернула некоректні поля класифікації.')
    return value


def validate_reply(value:dict)->dict:
    if set(value)!={'suggested_reply'}:
        raise SmartNotifyError('Модель повернула некоректні поля чернетки.')
    reply=value['suggested_reply']
    if reply is not None and (not isinstance(reply,str) or not reply.strip()):
        raise SmartNotifyError('Модель повернула некоректну чернетку.')
    return value


def classify(client:OpenAICompatibleClient,config:Config,sender:str,message:str):
    return_result,metrics=request_json(client,config,'classifier.txt',
         {'sender':sender,'message':message},CLASSIFIER_SCHEMA,'smartnotify_classifier')
    return validate_classifier(return_result),metrics


def generate_reply(client:OpenAICompatibleClient,config:Config,sender:str,message:str,urgency_level:str):
    if urgency_level not in LEVELS:
        raise SmartNotifyError('Невідома категорія терміновості.')
    result,metrics=request_json(client,config,'reply.txt',
         {'sender':sender,'message':message,'urgency_level':urgency_level},
         REPLY_SCHEMA,'smartnotify_reply')
    return validate_reply(result),metrics
