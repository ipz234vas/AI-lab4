"use strict";
const $ = id => document.getElementById(id);
const levels = {low:"Низька", medium:"Середня", high:"Висока", critical:"Критична"};
const contexts = [1024,1536,2048,3072,4096,8192];
let current = null;       // last successful analysis + optional reply: never erased by settings changes
let connection = null;    // {model, num_ctx} most recently connected and warmed
let knownConnections = new Set();
let busy = false;

function selection() {
  return {model:$('model').value, num_ctx:Number($('context').value)};
}
function settings() {
  return {...selection(), classifier_temperature:Number($('classifierTemperature').value),
    reply_temperature:Number($('replyTemperature').value), max_tokens:Number($('maxTokens').value),
    seed:$('seed').value.trim()==='' ? null : Number($('seed').value)};
}
function connectionKey({model,num_ctx}) {return JSON.stringify([model,num_ctx]);}
function isConnected() {const sel=selection();return !!sel.model && knownConnections.has(connectionKey(sel));}
function updateValues() {
  $('classifierTemperatureValue').textContent=Number($('classifierTemperature').value).toFixed(1);
  $('replyTemperatureValue').textContent=Number($('replyTemperature').value).toFixed(1);
  $('maxTokensValue').textContent=$('maxTokens').value;
}
function feedback(message, type='') {
  const el=$('feedback'); el.textContent=message;
  el.className=message ? `feedback ${type}` : 'feedback hidden';
}
function sameInput() {
  return !!current && current.input.sender===$('sender').value.trim() &&
    current.input.message===$('message').value.trim();
}
function updateConnectionStatus() {
  const el=$('serverStatus');
  if (!$('model').value) {
    el.className='server bad'; el.textContent='Немає доступної моделі';
  } else if (isConnected()) {
    el.className='server ok'; el.textContent='Модель підключена';
  } else {
    el.className='server muted'; el.textContent='Не підключено';
  }
}
function updateControls() {
  $('classifyBtn').disabled=busy || !isConnected();
  $('replyBtn').disabled=busy || !isConnected() || !sameInput();
  $('connectBtn').disabled=busy || !$('model').value;
  // Don't disable settings: changing them is harmless to already displayed outputs.
  // During an active request we lock them so that the request uses a coherent snapshot.
  for(const id of ['sender','message','autoReply','classifierTemperature',
                    'replyTemperature','context','maxTokens','seed','model','refreshModels'])
    $(id).disabled=busy;
  updateConnectionStatus();
}
function setBusy(value) {busy=value;updateControls();}
async function request(route,data) {
  const resp=await fetch(`/api/${route}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  let result;try{result=await resp.json();}catch{throw new Error('Сервер повернув некоректну відповідь.');}
  if(!resp.ok)throw new Error(result.error || 'Помилка сервера.');
  return result;
}
function validSeed() {
  const value=settings().seed;
  if (!$('seed').validity?.badInput && (value===null || (Number.isInteger(value) && value>=0 && value<=2147483647))) return true;
  feedback('Seed має бути цілим числом від 0 до 2147483647 або порожнім.','error');
  return false;
}
function displayMetrics() {
  if (!current)return;
  $('metricsText').textContent=JSON.stringify({classifier:current.classifier,reply:current.reply||null,
    connection:connection,latest_settings:settings()},null,2);
}
async function loadModels() {
  if (busy)return;
  const previous=$('model').value;
  $('refreshModels').disabled=true;
  try {
    const r=await fetch('/api/models');const data=await r.json();
    if(!r.ok)throw new Error(data.error||'Не вдалося отримати моделі Ollama.');
    const models=Array.isArray(data.models)?data.models:[];
    const sel=$('model');sel.replaceChildren();
    for(const name of models){const o=document.createElement('option');o.value=name;o.textContent=name;sel.appendChild(o);}
    if(models.includes(previous))sel.value=previous;
    else if(models.includes(data.preferred_model))sel.value=data.preferred_model;
    else if(models.length)sel.value=models[0];
    const supplied=Array.isArray(data.connected)?data.connected:[];
    knownConnections=new Set(supplied.map(connectionKey));
    if(connection && !knownConnections.has(connectionKey(connection)))connection=null;
    if(!models.length)feedback('У Ollama немає встановлених моделей. Встанови модель через ollama pull.','error');
    else feedback('');
  }catch(e){feedback(e.message,'error');}
  finally{$('refreshModels').disabled=false;updateControls();}
}
async function connectModel() {
  if(busy || !$('model').value)return;
  const selected=selection();setBusy(true);
  feedback(`Підключаю ${selected.model} · контекст ${selected.num_ctx}…`,'loading');
  try {
    const data=await request('connect',selected);
    knownConnections.add(connectionKey(selected));
    connection=selected;
    feedback(`Підключено. Прогрів: ${data.warmup.elapsed_s} с.`);
    // No changes to current, classifier, reply text or metrics.
    displayMetrics();
  } catch(e){feedback(e.message,'error');}
  finally{setBusy(false);}
}
async function createReply() {
  if(!current || busy && !$('autoReply').checked)return;
  if(!validSeed())return;
  if(!sameInput()) {feedback('Повідомлення змінилось. Натисни «Аналізувати» для нього.','error');return;}
  if(!isConnected()){feedback('Спочатку підключи вибрану модель та контекст.','error');return;}
  const alreadyBusy=busy;
  if(!alreadyBusy)setBusy(true);
  const params=settings();
  feedback('Створюю чернетку…','loading');
  try {
    const data=await request('reply',{
      ...current.input,model:params.model,num_ctx:params.num_ctx,max_tokens:params.max_tokens,
      temperature:params.reply_temperature,seed:params.seed,urgency_level:current.classifier.result.urgency_level
    });
    current.reply=data;
    $('reply').value=data.result.suggested_reply??'';
    $('reply').placeholder=data.result.suggested_reply===null?'Немає доречної чернетки':'';
    displayMetrics();
    feedback(data.result.suggested_reply===null?'Модель не запропонувала відповідь.':'');
  } catch(e){feedback(e.message,'error');}
  finally{if(!alreadyBusy)setBusy(false);}
}
async function analyze() {
  if(busy || !validSeed())return;
  if(!isConnected()){feedback('Спочатку натисни «Підключити» для вибраної моделі та контексту.','error');return;}
  const sender=$('sender').value.trim(),message=$('message').value.trim();
  if(!message){feedback('Введи повідомлення.','error');return;}
  const params=settings(); // snapshot at time of the request
  setBusy(true);feedback('Аналізую повідомлення…','loading');
  try {
    const input={sender,message};
    const cls=await request('classify',{...input,model:params.model,num_ctx:params.num_ctx,
      max_tokens:params.max_tokens,temperature:params.classifier_temperature,seed:params.seed});
    // Replace the old output ONLY AFTER successful new classification.
    current={input,settings_at_classification:params,classifier:cls,reply:null};
    const level=cls.result.urgency_level;
    $('urgency').className=`pill ${level}`;
    $('urgency').textContent=levels[level]||level;
    $('reason').textContent=cls.result.reason;
    $('reply').value='';
    $('reply').placeholder='Натисни «Згенерувати»';
    $('result').classList.remove('hidden');
    displayMetrics();feedback('');
    if($('autoReply').checked)await createReply();
  }catch(e){feedback(e.message,'error');}
  finally{setBusy(false);}
}
for(const id of ['classifierTemperature','replyTemperature','context','maxTokens','seed']){
  $(id).addEventListener('input',()=>{updateValues();updateControls();displayMetrics();});
  $(id).addEventListener('change',()=>{updateValues();updateControls();displayMetrics();});
}
$('model').addEventListener('change',()=>{updateControls();displayMetrics();});
// Editing the original message keeps the previous result visible, but the Reply
// button is disabled until that new message is classified (prevents stale drafts).
for(const id of ['sender','message'])$(id).addEventListener('input',updateControls);
$('connectBtn').addEventListener('click',connectModel);
$('refreshModels').addEventListener('click',loadModels);
$('classifyBtn').addEventListener('click',analyze);
$('replyBtn').addEventListener('click',()=>{if(!busy)return createReply();});
$('copyBtn').addEventListener('click',async()=>{
  if(!$('reply').value.trim())return feedback('Чернетка порожня.','error');
  try{await navigator.clipboard.writeText($('reply').value);feedback('Скопійовано.');}
  catch{feedback('Не вдалося скопіювати текст.','error');}
});
$('message').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')analyze();});
updateValues();updateControls();loadModels();
