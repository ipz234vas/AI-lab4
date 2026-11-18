/* Browser-state regression tests without npm dependencies: node app/tests/test_ui.js */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

class Element {
  constructor(id){this.id=id;this._value='';this.textContent='';this.checked=false;
    this.disabled=false;this.placeholder='';this.listeners={};this.children=[];
    const s=new Set(id==='result'||id==='feedback'?['hidden']:[]);
    this.classList={add:v=>s.add(v),remove:v=>s.delete(v),contains:v=>s.has(v)};
    this.className='';}
  get value(){return this._value;}
  set value(v){this._value=String(v);}
  addEventListener(type,cb){(this.listeners[type]??=[]).push(cb);}
  fire(type){return Promise.all((this.listeners[type]??[]).map(cb=>cb({ctrlKey:false,key:''})));}
  replaceChildren(){this.children=[];this._value='';}
  appendChild(el){this.children.push(el);if(this.children.length===1)this._value=el.value;}
}
const ids=['serverStatus','model','refreshModels','connectBtn','sender','message','classifyBtn','replyBtn','autoReply','classifierTemperature','replyTemperature','context','maxTokens','seed','classifierTemperatureValue','replyTemperatureValue','contextValue','maxTokensValue','feedback','result','urgency','reason','reply','metricsText','copyBtn'];
const elements=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
for(const [id,value] of Object.entries({context:'4096',maxTokens:'220',classifierTemperature:'0',replyTemperature:'0',sender:'Колега',message:'Привіт'}))elements[id].value=value;
let calls=[];
const document={getElementById(id){if(!elements[id])throw Error('missing '+id);return elements[id];},createElement(){return new Element('option');}};
const response=(payload,status=200)=>({ok:status<400,async json(){return payload;}});
async function fetch(url,options={}){
  const body=options.body?JSON.parse(options.body):null;
  calls.push({url,body});
  if(url==='/api/models')return response({models:['qwen3.5:2b-q4_K_M','qwen3:4b-q4_K_M'],connected:[],preferred_model:'qwen3.5:2b-q4_K_M'});
  if(url==='/api/connect')return response({connected:true,warmup:{elapsed_s:1.4}});
  if(url==='/api/classify')return response({result:{urgency_level:'high',reason:'Потрібно відповісти.'},metrics:{elapsed_s:2},settings:body});
  if(url==='/api/reply')return response({result:{suggested_reply:'Чернетка: '+body.temperature},metrics:{elapsed_s:1},settings:body});
  throw Error('Unexpected '+url);
}
const ctx={document,fetch,navigator:{clipboard:{writeText:async()=>{}}},console};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8'),ctx);
const tick=()=>new Promise(r=>setImmediate(r));
async function run(){
  await tick();await tick();
  assert.equal(elements.model.value,'qwen3.5:2b-q4_K_M');
  assert.equal(elements.classifyBtn.disabled,true,'requires connect before first request');
  await elements.connectBtn.fire('click');
  assert.equal(elements.classifyBtn.disabled,false);
  await elements.classifyBtn.fire('click');
  assert.equal(elements.result.classList.contains('hidden'),false);
  assert.equal(elements.urgency.textContent,'Висока');
  await elements.replyBtn.fire('click');
  assert.equal(elements.reply.value,'Чернетка: 0');
  assert.equal(calls.filter(c=>c.url==='/api/classify').at(-1).body.seed,null);
  assert.equal(calls.filter(c=>c.url==='/api/reply').at(-1).body.seed,null);
  const originalReply=elements.reply.value;
  elements.seed.value='0';await elements.seed.fire('input');
  assert.equal(elements.reply.value,originalReply);
  const origClassCalls=calls.filter(c=>c.url==='/api/classify').length;
  // BOTH temperatures and max tokens should leave existing output intact.
  elements.replyTemperature.value='0.5';await elements.replyTemperature.fire('input');
  elements.classifierTemperature.value='0.3';await elements.classifierTemperature.fire('change');
  elements.maxTokens.value='128';await elements.maxTokens.fire('input');
  assert.equal(elements.reply.value,originalReply);
  assert.equal(elements.result.classList.contains('hidden'),false);
  assert.equal(elements.urgency.textContent,'Висока');
  await elements.replyBtn.fire('click');
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  assert.equal(calls.filter(c=>c.url==='/api/classify').length,origClassCalls,
    'reply must not rerun classification');
  assert.equal(calls.filter(c=>c.url==='/api/reply').at(-1).body.max_tokens,128);
  assert.equal(calls.filter(c=>c.url==='/api/reply').at(-1).body.seed,0);
  elements.seed.value='123';await elements.seed.fire('input');
  const oldText=elements.reply.value;
  elements.seed.value='-1';await elements.seed.fire('input');
  const count=calls.length;await elements.replyBtn.fire('click');
  assert.equal(calls.length,count);assert.equal(elements.reply.value,oldText);
  elements.seed.value='123';await elements.seed.fire('input');
  // Switching model keeps results visible and requires warming new model.
  elements.model.value='qwen3:4b-q4_K_M';await elements.model.fire('change');
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  assert.equal(elements.classifyBtn.disabled,true);
  await elements.connectBtn.fire('click');
  assert.equal(elements.classifyBtn.disabled,false);
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  await elements.replyBtn.fire('click');
  assert.equal(calls.filter(c=>c.url==='/api/reply').at(-1).body.model,'qwen3:4b-q4_K_M');
  // Changing ctx requires a new warmup; does not clear anything.
  elements.context.value='2048';await elements.context.fire('change');
  assert.equal(elements.classifyBtn.disabled,true);
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  await elements.connectBtn.fire('click');
  assert.equal(elements.classifyBtn.disabled,false);
  assert.equal(calls.filter(c=>c.url==='/api/connect').at(-1).body.num_ctx,2048);
  elements.context.value='1024';await elements.context.fire('change');
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  assert.equal(elements.classifyBtn.disabled,true);
  await elements.connectBtn.fire('click');
  assert.equal(calls.filter(c=>c.url==='/api/connect').at(-1).body.num_ctx,1024);
  // Editing input keeps old result but won't let us reply to stale input.
  elements.message.value='Змінене повідомлення';await elements.message.fire('input');
  assert.equal(elements.result.classList.contains('hidden'),false);
  assert.equal(elements.replyBtn.disabled,true);
  await elements.classifyBtn.fire('click');
  assert.equal(elements.reply.value,'');
  assert.equal(elements.replyBtn.disabled,false);
  // Auto-reply is still a second independent call after successful classification.
  elements.autoReply.checked=true;
  const oldReplies=calls.filter(c=>c.url==='/api/reply').length;
  await elements.classifyBtn.fire('click');
  assert.equal(calls.filter(c=>c.url==='/api/reply').length,oldReplies+1);
  assert.equal(elements.reply.value,'Чернетка: 0.5');
  assert.equal(calls.filter(c=>c.url==='/api/classify').at(-1).body.seed,123);
  assert.equal(calls.filter(c=>c.url==='/api/reply').at(-1).body.seed,123);
  console.log('UI regression tests passed: temperatures, model switch, context, warmup, result persistence, auto reply.');
}
run().catch(e=>{console.error(e);process.exitCode=1;});
