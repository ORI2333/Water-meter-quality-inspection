#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chinese web dashboard for FPGA HDMI YOLO11-pose water-meter detection."""
import argparse
import json
import math
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
from rknnlite.api import RKNNLite

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import usb_rknn_detect as det
import hdmi_yolo11_pose_detect as pose
from hdmi_rknn_detect import HDMIFrameSource

CONFIG_PATH = "/home/demo/water_meter/water_meter_web_config.json"
FP_MODEL_PATH = "/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn"
FAST_MODEL_PATH = "/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn"
DIAL_LABELS = ["10^-1", "10^-2", "10^-3", "10^-4"]
DIAL_WEIGHTS = [0.1, 0.01, 0.001, 0.0001]
DIAL_VOLUME_PER_TURN = [w * 10.0 for w in DIAL_WEIGHTS]
PRIMARY_VOLUME_DIAL = 3
DEFAULT_CONFIG = {
    "zero_offsets": [0.0, 0.0, 0.0, 0.0],
    "directions": [1, -1, 1, -1],
    "base_m3": 0.0,
    "measurement_source": "auto",
}


def model_mode_from_path(model_path):
    return "fast" if os.path.basename(str(model_path)) == os.path.basename(FAST_MODEL_PATH) else "accuracy"

HTML_PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>水表指针检测 Web 仪表盘</title>
<style>
:root{color-scheme:light;--bg:#f6f7f9;--surface:#fff;--surface2:#f0f3f6;--line:#e4e7eb;--line2:#cfd6df;--text:#18191c;--muted:#68707c;--soft:#3f4752;--accent:#00aeec;--accent2:#00a65a;--warn:#d98700;--bad:#d9363e;--shadow:0 10px 30px rgba(21,31,46,.10)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Microsoft YaHei",Inter,"Segoe UI",Arial,sans-serif;letter-spacing:0}
button,input{font:inherit}button{height:34px;border:1px solid var(--line2);background:#fff;color:#20242a;border-radius:6px;font-weight:650;cursor:pointer;transition:.16s border-color,.16s background,.16s color,.16s transform}button:hover{border-color:var(--accent);color:#0077a3;background:#f2fbff}button:active{transform:translateY(1px)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}.ghost{background:#fff}.danger{border-color:#ffb5ba;color:#b4232c;background:#fff6f7}
input{height:36px;border-radius:6px;border:1px solid var(--line2);background:#fff;color:#111;padding:0 10px;min-width:0}.appbar{height:58px;display:grid;grid-template-columns:240px minmax(260px,1fr) auto;gap:18px;align-items:center;padding:0 20px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.96);position:sticky;top:0;z-index:5;backdrop-filter:blur(10px)}
.brand{display:flex;align-items:center;gap:10px;min-width:0}.brand-mark{width:32px;height:32px;border-radius:8px;background:#00aeec;color:#fff;display:grid;place-items:center;font-weight:900}.brand-text{min-width:0}.brand h1{font-size:16px;margin:0;font-weight:800;white-space:nowrap}.sub{color:var(--muted);font-size:12px;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status-strip{height:36px;border:1px solid var(--line);background:var(--surface2);border-radius:18px;display:flex;align-items:center;gap:8px;padding:0 10px;min-width:0}.head-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}.pill{border:1px solid var(--line2);padding:6px 10px;border-radius:999px;color:var(--soft);font-size:12px;background:#fff;white-space:nowrap}.pill.ok{color:#087a41;border-color:#b7e4cc;background:#f1fff7}.pill.bad{color:#b4232c;border-color:#ffc2c7;background:#fff5f6}.pill.warn{color:#9a6500;border-color:#f4d08a;background:#fff9e8}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(580px,640px);gap:18px;padding:18px;min-height:calc(100vh - 58px);max-width:1820px;margin:0 auto}.stage{display:flex;flex-direction:column;gap:12px;min-width:0;width:100%}
.video{width:100%;max-width:var(--player-width,100%);align-self:center;background:#030303;border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;aspect-ratio:16/9;min-height:320px;box-shadow:var(--shadow);position:relative;transition:.18s max-width}.stage.size-small .video{max-width:760px;min-height:240px}.stage.size-medium .video{max-width:1040px;min-height:300px}.stage.size-large .video{max-width:100%}.stage.size-custom .video{max-width:var(--player-width,1180px)}.video img{width:100%;height:100%;object-fit:contain;display:block}.video.paused:after{content:"已暂停";position:absolute;right:14px;top:14px;background:rgba(0,0,0,.72);border:1px solid rgba(255,255,255,.22);border-radius:6px;padding:7px 11px;color:#fff;font-weight:700}
.watch-meta{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;box-shadow:0 4px 18px rgba(21,31,46,.05)}.title-line{display:flex;gap:16px;align-items:flex-start;justify-content:space-between}.video-title{font-size:18px;font-weight:800;margin:0 0 5px}.reading-block{text-align:right;min-width:190px}.reading-label{font-size:12px;color:var(--muted)}.reading{font-size:30px;font-weight:820;font-variant-numeric:tabular-nums;color:#00a1d6;line-height:1.15;margin-top:2px}
.control-strip{display:flex;align-items:center;justify-content:space-between;gap:10px;border-top:1px solid var(--line);padding-top:12px;margin-top:12px;flex-wrap:wrap}.control-left,.control-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.toggle{display:inline-flex;gap:7px;align-items:center;color:var(--soft);font-size:13px}.toggle input{height:auto}.seg{display:inline-flex;border:1px solid var(--line2);border-radius:7px;overflow:hidden;background:#fff}.seg button{border:0;border-right:1px solid var(--line2);border-radius:0;background:#fff;height:32px}.seg button:last-child{border-right:0}.seg button.active{background:#e6f7ff;color:#0077a3}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:8px;margin-top:12px}.stat{display:flex;align-items:center;justify-content:space-between;gap:8px;border:1px solid var(--line);background:#fafbfc;border-radius:6px;padding:9px 10px;min-height:42px}.stat span:first-child{font-size:12px;color:var(--muted)}.stat strong{font-size:17px;font-variant-numeric:tabular-nums}.stat.ok strong{color:var(--accent2)}.stat.accent strong{color:#00a1d6}
.reading-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.mini{border:1px solid var(--line);background:#fafbfc;border-radius:6px;padding:9px}.mini .label{font-size:12px;color:var(--muted)}.mini .num{display:block;margin-top:3px;color:#18191c}.side{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;align-content:start;min-width:0}.rail-title{font-size:14px;font-weight:800;color:#18191c;margin:0 0 10px}.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 4px 18px rgba(21,31,46,.05)}.card.wide{grid-column:1/-1}.card h2{font-size:14px;margin:0 0 12px;color:#18191c;font-weight:800}.small{font-size:13px;color:var(--muted);line-height:1.58}
.tuning{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin-top:12px;border-top:1px solid var(--line);padding-top:12px}.tune{border:1px solid var(--line);background:#fafbfc;border-radius:6px;padding:9px 10px}.tune label{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px}.tune b{color:#18191c;font-variant-numeric:tabular-nums}.tune input[type=range]{width:100%;height:22px;margin:6px 0 0;accent-color:var(--accent)}
.dials{display:grid;grid-template-columns:1fr 1fr;gap:8px}.dial{border:1px solid var(--line);border-radius:7px;padding:10px;background:#fff}.dial-top,.row{display:flex;justify-content:space-between;gap:8px;align-items:center}.dial-name{font-weight:800;color:#18191c}.score{font-size:12px;color:var(--muted)}.row{font-size:13px;color:#68707c;margin-top:7px}.num{font-variant-numeric:tabular-nums;color:#18191c}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.base-row{margin-top:10px;display:flex;gap:8px}.base-row button{width:96px}.source-row{margin-top:10px}.source-row .seg{width:100%;display:flex}.source-row .seg button{flex:1;padding:0 6px}.source-title{font-size:12px;color:var(--muted);margin-bottom:6px}.msg{min-height:18px;margin-top:10px}.kv{display:grid;grid-template-columns:72px minmax(0,1fr);gap:6px 10px}.kv div:nth-child(odd){color:var(--muted)}.kv div:nth-child(even){color:#18191c;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.cmd{font-family:Consolas,monospace;font-size:12px;color:#475467;background:#f6f7f9;border-radius:6px;padding:10px;white-space:pre-wrap}.footer{font-size:12px;color:var(--muted);line-height:1.6}
@media(max-width:1180px){.appbar{grid-template-columns:220px 1fr}.head-right{grid-column:1/-1;justify-content:flex-start;margin-bottom:10px}.appbar{height:auto;padding:10px 14px}main{grid-template-columns:1fr}.side{order:2}.tuning{grid-template-columns:repeat(2,minmax(150px,1fr))}.video{min-height:260px}}
@media(max-width:720px){main{padding:10px}.appbar{grid-template-columns:1fr}.status-strip{order:3}.title-line{flex-direction:column}.reading-block{text-align:left}.side,.dials,.metrics,.tuning{grid-template-columns:1fr}.control-strip{align-items:flex-start}.reading{font-size:27px}}
</style>
</head>
<body>
<header class="appbar">
  <div class="brand"><div class="brand-mark">WM</div><div class="brand-text"><h1>水表检测台</h1><div class="sub">FPGA HDMI 实时视频</div></div></div>
  <div class="status-strip"><span id="health" class="pill">启动中</span><span id="pauseState" class="pill">实时</span><span id="url" class="pill"></span></div>
  <div class="head-right"><span class="pill">YOLO11-pose</span><span class="pill">RK3588 NPU</span></div>
</header>
<main>
  <section id="stage" class="stage size-large">
    <section id="videoBox" class="video"><img id="stream" src="/stream" alt="视频流"></section>
    <section class="watch-meta">
      <div class="title-line">
        <div><h2 class="video-title">HDMI 输入实时识别</h2><div class="sub">左侧显示 FPGA 输出画面，检测标注可开关；精度模式适合点位观察，快速模式适合高帧率巡检。</div></div>
        <div class="reading-block"><div class="reading-label">本次经过水量</div><div id="reading" class="reading">-- m³</div></div>
      </div>
      <div class="control-strip">
        <div class="control-left">
          <button id="pauseBtn" class="primary" onclick="togglePause()">暂停</button>
          <button class="ghost" onclick="reloadStream()">刷新视频</button>
          <button class="ghost" onclick="openSnapshot()">打开截图</button>
          <label class="toggle"><input id="overlayToggle" type="checkbox" checked onchange="setOverlay(this.checked)">显示检测标注</label>
        </div>
        <div class="control-right">
          <span class="seg"><button id="modeAccuracy" onclick="switchModelMode('accuracy')">精度 FP</button><button id="modeFast" onclick="switchModelMode('fast')">快速 INT8</button></span>
          <span class="seg"><button id="sizeSmall" onclick="setVideoSize('small')">小窗</button><button id="sizeMedium" onclick="setVideoSize('medium')">中窗</button><button id="sizeLarge" onclick="setVideoSize('large')">影院</button></span>
        </div>
      </div>
      <div class="metrics">
        <div class="stat ok"><span>检测</span><strong id="det">--</strong></div>
        <div class="stat accent"><span>推理 FPS</span><strong id="fps">--</strong></div>
        <div class="stat"><span>推理耗时</span><strong id="infer">--</strong></div>
        <div class="stat"><span>网页 FPS</span><strong id="displayFps">--</strong></div>
        <span id="frames" hidden>--</span>
      </div>
      <div class="reading-row">
        <div class="mini"><div class="label">估算总读数</div><span id="decimalM3" class="num">--</span></div>
        <div class="mini"><div class="label">测量来源</div><span id="cumReading" class="num">--</span></div>
      </div>
      <div class="tuning">
        <div class="tune"><label>视频框宽度 <b id="playerWidthText">--</b></label><input id="playerWidth" type="range" min="640" max="1500" step="20" oninput="setPlayerWidth(this.value)"></div>
        <div class="tune"><label>推流宽度 <b id="streamWidthText">--</b></label><input id="streamWidth" type="range" min="480" max="1280" step="80" oninput="queueStreamSettings()"></div>
        <div class="tune"><label>JPEG 质量 <b id="qualityText">--</b></label><input id="streamQuality" type="range" min="45" max="98" step="1" oninput="queueStreamSettings()"></div>
        <div class="tune"><label>网页帧率 <b id="streamFpsText">--</b></label><input id="streamFps" type="range" min="2" max="30" step="1" oninput="queueStreamSettings()"></div>
      </div>
    </section>
  </section>
  <aside class="side">
    <section class="card"><h2>校准与读数</h2><div class="actions"><button onclick="postAction('reset_turns')">开始测量/清零</button><button onclick="postAction('calibrate_zero')">当前设零位</button></div><div class="base-row"><input id="baseM3" type="number" step="0.001" min="0" placeholder="起始/基准 m³"><button onclick="setBaseM3()">保存</button></div><div class="source-row"><div class="source-title">测量来源</div><span class="seg"><button id="srcAuto" onclick="setMeasureSource('auto')">自动</button><button id="src0" onclick="setMeasureSource('0')">10^-1</button><button id="src1" onclick="setMeasureSource('1')">10^-2</button><button id="src2" onclick="setMeasureSource('2')">10^-3</button><button id="src3" onclick="setMeasureSource('3')">10^-4</button></span></div><div id="actionMsg" class="small msg"></div></section>
    <section class="card"><h2>输入与模型</h2><div id="input" class="kv small">--</div></section>
    <section class="card wide"><h2>表盘检测列表</h2><div id="dials" class="dials small">等待检测结果</div></section>
    <section class="card"><h2>启动命令</h2><div class="cmd">精度优先：./run_hdmi_yolo11_pose_web.sh
速度优先：WM_MODEL_MODE=fast ./run_hdmi_yolo11_pose_web.sh</div></section>
    <section class="card footer">视频流：/stream<br>状态接口：/status<br>控制接口：/control</section>
  </aside>
</main>
<script>
const $=id=>document.getElementById(id); $('url').textContent=location.host;
let lastStatus={};
let streamTimer=null;
function fmt(v,d=1){return Number.isFinite(v)?v.toFixed(d):'--'}
async function postControl(payload){
  const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const s=await r.json(); if(!r.ok||s.ok===false) throw new Error(s.message||'操作失败'); return s;
}
async function postAction(action){
  try{const s=await postControl({action}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='操作失败：'+e.message;}
}
async function togglePause(){
  try{const s=await postControl({action:'set_pause',paused:!lastStatus.paused}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='暂停控制失败：'+e.message;}
}
async function setOverlay(enabled){
  try{const s=await postControl({action:'set_overlay',overlay:enabled}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='标注控制失败：'+e.message;}
}
function reloadStream(){const img=$('stream'); img.src='/stream?ts='+Date.now();}
function openSnapshot(){window.open('/snapshot.jpg?ts='+Date.now(),'_blank');}
function setVideoSize(size){
  localStorage.setItem('wmVideoSize',size);
  $('stage').className='stage size-'+size;
  ['Small','Medium','Large'].forEach(n=>$('size'+n).classList.toggle('active', n.toLowerCase()===size));
  const preset={small:760,medium:1040,large:1500}[size]||1180;
  document.documentElement.style.setProperty('--player-width',preset+'px');
  $('playerWidth').value=preset; $('playerWidthText').textContent=preset+' px';
}
function setPlayerWidth(value,save=true){
  const w=Math.max(640,Math.min(1500,parseInt(value||1180,10)));
  document.documentElement.style.setProperty('--player-width',w+'px');
  $('stage').className='stage size-custom';
  $('playerWidth').value=w; $('playerWidthText').textContent=w+' px';
  ['Small','Medium','Large'].forEach(n=>$('size'+n).classList.remove('active'));
  if(save!==false){localStorage.setItem('wmVideoSize','custom'); localStorage.setItem('wmPlayerWidth',String(w));}
}
function queueStreamSettings(){
  const sw=parseInt($('streamWidth').value,10), q=parseInt($('streamQuality').value,10), fps=parseInt($('streamFps').value,10);
  $('streamWidthText').textContent=sw+' px'; $('qualityText').textContent=q; $('streamFpsText').textContent=fps+' fps';
  clearTimeout(streamTimer);
  streamTimer=setTimeout(async()=>{
    try{
      const s=await postControl({action:'set_stream',stream_width:sw,jpeg_quality:q,stream_fps:fps});
      $('actionMsg').textContent=s.message||'推流参数已更新';
      reloadStream();
    }catch(e){$('actionMsg').textContent='推流参数设置失败：'+e.message;}
  },300);
}
async function switchModelMode(mode){
  const label=mode==='fast'?'快速 INT8':'精度 FP';
  if(!confirm('切换到 '+label+' 会自动重启 Web 服务，约 5 秒后恢复。继续？')) return;
  try{const s=await postControl({action:'restart_model_mode',mode}); $('actionMsg').textContent=s.message||'正在重启'; setTimeout(()=>location.reload(),5500);}
  catch(e){$('actionMsg').textContent='切换失败：'+e.message;}
}
async function setBaseM3(){
  const v=parseFloat($('baseM3').value);
  if(!Number.isFinite(v)){ $('actionMsg').textContent='请输入有效的 m³ 基值'; return; }
  try{const s=await postControl({action:'set_base_m3',base_m3:v}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='保存失败：'+e.message;}
}
async function setMeasureSource(src){
  try{const s=await postControl({action:'set_measurement_source',source:src}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='测量来源设置失败：'+e.message;}
}
async function setDialDirection(idx,direction){
  try{const s=await postControl({action:'set_dial_direction',idx,direction}); $('actionMsg').textContent=s.message||'完成'; tick();}
  catch(e){$('actionMsg').textContent='方向设置失败：'+e.message;}
}
function dialHtml(d){const dir=d.direction>=0?1:-1; const dirText=dir>0?'正向':'反向'; const next=-dir; return `<div class="dial"><div class="dial-top"><span class="dial-name">${d.label}</span><span class="score">置信度 ${fmt(d.score,2)}</span></div><div class="row"><span>原始/稳定角</span><span class="num">${fmt(d.raw_angle,1)}° / ${fmt(d.stable_angle,1)}°</span></div><div class="row"><span>刻度</span><span class="num">${fmt(d.digit,2)} / 10</span></div><div class="row"><span>累计圈数</span><span class="num">${fmt(d.turns,3)}</span></div><div class="row"><span>本表估算水量</span><span class="num">${fmt(d.delta_m3,6)} m³</span></div><div class="row"><span>每圈水量</span><span class="num">${fmt(d.volume_per_turn,4)} m³</span></div><div class="row"><span>方向</span><span><button class="ghost" onclick="setDialDirection(${d.idx},${next})">${dirText}</button></span></div></div>`}
function setHealth(s){
  const stale=Number.isFinite(s.updated)&&(Date.now()/1000-s.updated)>3;
  $('health').textContent=s.error?'异常':(stale?'等待数据':'运行中');
  $('health').className='pill '+(s.error?'bad':(stale?'warn':'ok'));
  $('pauseState').textContent=s.paused?'已暂停':'实时';
  $('pauseState').className='pill '+(s.paused?'warn':'ok');
  $('pauseBtn').textContent=s.paused?'继续':'暂停';
  $('videoBox').className='video '+(s.paused?'paused':'');
  $('modeAccuracy').classList.toggle('active', s.model_mode!=='fast');
  $('modeFast').classList.toggle('active', s.model_mode==='fast');
}
async function tick(){
  try{
    const r=await fetch('/status',{cache:'no-store'}); const s=await r.json(); lastStatus=s; setHealth(s);
    $('fps').textContent=fmt(s.fps,1); $('displayFps').textContent=fmt(s.display_fps,1); $('infer').textContent=fmt(s.infer_ms,1)+' ms'; $('det').textContent=s.det_count??'--'; $('frames').textContent=s.frame_count??'--';
    $('reading').textContent=fmt(s.elapsed_m3,6)+' m³'; $('decimalM3').textContent=fmt(s.total_m3,6)+' m³'; $('cumReading').textContent=s.measurement_source||'--';
    if(document.activeElement!==$('baseM3')) $('baseM3').value=Number.isFinite(s.base_m3)?s.base_m3:'';
    if(document.activeElement!==$('overlayToggle')) $('overlayToggle').checked=s.overlay_enabled!==false;
    if(document.activeElement!==$('streamWidth')) {$('streamWidth').value=s.stream_width||1280; $('streamWidthText').textContent=(s.stream_width||0)+' px';}
    if(document.activeElement!==$('streamQuality')) {$('streamQuality').value=s.jpeg_quality||90; $('qualityText').textContent=s.jpeg_quality||'--';}
    if(document.activeElement!==$('streamFps')) {$('streamFps').value=s.stream_fps||12; $('streamFpsText').textContent=(s.stream_fps||0)+' fps';}
    const cfgSrc=String(s.measurement_source_config||'auto');
    ['Auto','0','1','2','3'].forEach(n=>{const el=$('src'+n); if(el) el.classList.toggle('active',(n==='Auto'?'auto':n)===cfgSrc);});
    $('input').innerHTML=`<div>模式</div><div>${s.model_mode==='fast'?'快速 INT8':'精度 FP'}</div><div>设备</div><div>${s.device||''}</div><div>输入</div><div>${s.width||0}x${s.height||0}@${s.req_fps||0}</div><div>模型</div><div title="${s.model||''}">${s.model||''}</div><div>NPU</div><div>${s.core_mask||''}</div><div>网页流</div><div>${s.stream_width||0}px / JPEG ${s.jpeg_quality||''} / ${s.stream_fps||0}fps</div><div>状态</div><div>${s.error||'正常'}</div>`;
    if(s.dials&&s.dials.length){$('dials').innerHTML=s.dials.map(dialHtml).join('');} else {$('dials').textContent='等待检测结果';}
  }catch(e){$('health').textContent='离线';$('health').className='pill bad';}
}
const savedSize=localStorage.getItem('wmVideoSize')||'large';
if(savedSize==='custom') setPlayerWidth(localStorage.getItem('wmPlayerWidth')||1180,false); else setVideoSize(savedSize);
setInterval(tick,500); tick();
</script>
</body>
</html>
'''


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            disk = json.load(f)
        for key in cfg:
            if key in disk:
                cfg[key] = disk[key]
    except Exception:
        pass
    cfg["zero_offsets"] = (list(cfg.get("zero_offsets", [])) + [0.0] * 4)[:4]
    cfg["directions"] = (list(cfg.get("directions", [])) + [1, -1, 1, -1])[:4]
    cfg["directions"] = [1 if int(v) >= 0 else -1 for v in cfg["directions"]]
    try:
        cfg["base_m3"] = float(cfg.get("base_m3", 0.0))
    except Exception:
        cfg["base_m3"] = 0.0
    src = str(cfg.get("measurement_source", "auto")).strip().lower()
    cfg["measurement_source"] = src if src in ("auto", "0", "1", "2", "3") else "auto"
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


def parse_args():
    p = argparse.ArgumentParser(description="Local web dashboard for HDMI YOLO11-pose detection")
    p.add_argument("--model", default="/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn")
    p.add_argument("--device", default="/dev/video73")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--input-width", type=int, default=640)
    p.add_argument("--input-height", type=int, default=640)
    p.add_argument("--input-layout", default="nhwc", choices=("auto", "nchw", "nhwc"))
    p.add_argument("--input-dtype", default="uint8", choices=("auto", "uint8", "int8", "float32"))
    p.add_argument("--color", default="rgb", choices=("bgr", "rgb"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--angle-deadband", type=float, default=1.0)
    p.add_argument("--angle-alpha", type=float, default=0.30)
    p.add_argument("--angle-confirm-frames", type=int, default=4)
    p.add_argument("--angle-confirm-band", type=float, default=2.0)
    p.add_argument("--turn-deadband", type=float, default=4.0)
    p.add_argument("--center-deadband", type=float, default=6.0)
    p.add_argument("--radius-deadband", type=float, default=5.0)
    p.add_argument("--stream-width", type=int, default=800)
    p.add_argument("--stream-fps", type=float, default=12.0)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--infer-every", type=int, default=1)
    p.add_argument("--core-mask", default="all", choices=("auto", "all", "0", "1", "2", "01", "012"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=6008)
    p.add_argument("--print-every", type=int, default=60)
    return p.parse_args()


def core_mask_value(name):
    return {
        "auto": RKNNLite.NPU_CORE_AUTO,
        "all": RKNNLite.NPU_CORE_0_1_2,
        "012": RKNNLite.NPU_CORE_0_1_2,
        "01": RKNNLite.NPU_CORE_0_1,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
    }[name]


def circular_diff(new_angle, old_angle):
    return (new_angle - old_angle + 180.0) % 360.0 - 180.0


class LockedAngleStabilizer:
    def __init__(self, deadband_deg=3.0, alpha=0.25, confirm_frames=4, confirm_band_deg=2.0):
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.alpha = min(1.0, max(0.01, float(alpha)))
        self.confirm_frames = max(1, int(confirm_frames))
        self.confirm_band_deg = max(0.1, float(confirm_band_deg))
        self.state = {}

    def update(self, key, raw_angle):
        raw_angle = float(raw_angle) % 360.0
        st = self.state.get(key)
        if st is None:
            self.state[key] = {"stable": raw_angle, "candidate": None, "count": 0}
            return raw_angle

        stable = st["stable"]
        diff = circular_diff(raw_angle, stable)
        if abs(diff) <= self.deadband_deg:
            st["candidate"] = None
            st["count"] = 0
            return stable

        candidate = st["candidate"]
        if candidate is None or abs(circular_diff(raw_angle, candidate)) > self.confirm_band_deg:
            st["candidate"] = raw_angle
            st["count"] = 1
            return stable

        candidate = (candidate + 0.5 * circular_diff(raw_angle, candidate)) % 360.0
        st["candidate"] = candidate
        st["count"] += 1
        if st["count"] >= self.confirm_frames:
            stable = (stable + self.alpha * circular_diff(candidate, stable)) % 360.0
            st["stable"] = stable
            if abs(circular_diff(raw_angle, stable)) <= self.deadband_deg:
                st["candidate"] = None
                st["count"] = 0
        return stable


class GeometryStabilizer:
    def __init__(self, center_deadband=6.0, radius_deadband=5.0, alpha=0.18):
        self.center_deadband = max(0.0, float(center_deadband))
        self.radius_deadband = max(0.0, float(radius_deadband))
        self.alpha = min(1.0, max(0.01, float(alpha)))
        self.state = {}

    def update(self, key, center, radius):
        center = np.asarray(center, dtype=np.float32)
        radius = float(radius)
        st = self.state.get(key)
        if st is None:
            self.state[key] = {"center": center.copy(), "radius": radius}
            return center, radius

        stable_center = st["center"]
        stable_radius = st["radius"]
        dist = float(np.linalg.norm(center - stable_center))
        if dist > self.center_deadband:
            stable_center = stable_center + self.alpha * (center - stable_center)
            st["center"] = stable_center
        if abs(radius - stable_radius) > self.radius_deadband:
            stable_radius = stable_radius + self.alpha * (radius - stable_radius)
            st["radius"] = stable_radius
        return stable_center, stable_radius


def logical_angle(angle, zero_offset, direction):
    return ((angle - zero_offset) * direction) % 360.0


def keep_best_per_class(dets):
    best = {}
    for d in dets:
        old = best.get(d.cls_id)
        if old is None or d.score > old.score:
            best[d.cls_id] = d
    return [best[k] for k in sorted(best.keys())]


class TurnCounter:
    def __init__(self, config, turn_deadband_deg=4.0):
        self.config = config
        self.turn_deadband_deg = max(0.0, float(turn_deadband_deg))
        self.prev_logic = {}
        self.turns = {i: 0.0 for i in range(4)}
        self.last_angles = {}

    def reset_turns(self):
        self.prev_logic.clear()
        self.turns = {i: 0.0 for i in range(4)}

    def calibrate_zero(self):
        for i in range(4):
            if i in self.last_angles:
                self.config["zero_offsets"][i] = float(self.last_angles[i])
        self.reset_turns()
        save_config(self.config)

    def update(self, cls_id, angle, score):
        idx = int(cls_id)
        if idx < 0 or idx >= 4:
            return None
        zero = float(self.config["zero_offsets"][idx])
        direction = int(self.config["directions"][idx])
        logic = logical_angle(angle, zero, direction)
        if idx in self.prev_logic:
            diff = circular_diff(logic, self.prev_logic[idx])
            if abs(diff) >= self.turn_deadband_deg:
                self.turns[idx] += diff / 360.0
                self.prev_logic[idx] = logic
        else:
            self.prev_logic[idx] = logic
        self.last_angles[idx] = float(angle)
        digit = logic / 36.0
        reading_part = digit * DIAL_WEIGHTS[idx]
        delta_m3 = self.turns[idx] * DIAL_VOLUME_PER_TURN[idx]
        return {
            "idx": idx,
            "label": DIAL_LABELS[idx],
            "angle": float(angle),
            "logic_angle": float(logic),
            "digit": float(digit),
            "turns": float(self.turns[idx]),
            "weight": DIAL_WEIGHTS[idx],
            "volume_per_turn": DIAL_VOLUME_PER_TURN[idx],
            "reading_part": float(reading_part),
            "delta_m3": float(delta_m3),
            "score": float(score),
            "direction": direction,
            "zero_offset": zero,
        }

    def elapsed_m3(self, dials, source="auto"):
        present = {int(d.get("idx", -1)): d for d in dials}
        if not present:
            return 0.0, "--"
        source = str(source or "auto").strip().lower()
        if source in ("0", "1", "2", "3") and int(source) in present:
            idx = int(source)
            label = DIAL_LABELS[idx]
        else:
            # For water-throughput tests, prefer the most sensitive dial that
            # has accumulated positive motion. A negative value usually means
            # that dial's configured direction or keypoint is unreliable.
            moving = [idx for idx in sorted(present.keys(), reverse=True) if float(self.turns.get(idx, 0.0)) >= 0.005]
            idx = moving[0] if moving else max(present.keys())
            label = "自动:" + DIAL_LABELS[idx]
        return float(self.turns.get(idx, 0.0)) * DIAL_VOLUME_PER_TURN[idx], label


def draw_pose_stable(frame, dets, stabilizer, counter, geometry=None, overlay_enabled=True):
    dials = []
    for d in dets:
        color = pose.COLORS[d.cls_id % len(pose.COLORS)]
        x1, y1, x2, y2 = d.box.astype(int).tolist()
        center = d.kpts[0, :2]
        tip = d.kpts[1, :2]
        raw_angle = pose.pointer_angle_deg(center, tip)
        angle = stabilizer.update(d.cls_id, raw_angle) if stabilizer is not None else raw_angle
        if overlay_enabled:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, tuple(center.astype(int)), 4, (0, 255, 255), -1)
            cv2.circle(frame, tuple(tip.astype(int)), 5, (0, 0, 255), -1)
            cv2.line(frame, tuple(center.astype(int)), tuple(tip.astype(int)), (255, 0, 0), 2)
        dial = counter.update(d.cls_id, raw_angle, d.score)
        if dial is not None:
            dial["raw_angle"] = float(raw_angle)
            dial["stable_angle"] = float(angle)
            dials.append(dial)
        if overlay_enabled:
            label = DIAL_LABELS[d.cls_id] if d.cls_id < len(DIAL_LABELS) else str(d.cls_id)
            cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    dials.sort(key=lambda x: DIAL_LABELS.index(x["label"]) if x["label"] in DIAL_LABELS else 99)
    return dials


def update_dials_only(dets, stabilizer, counter):
    dials = []
    for d in dets:
        center = d.kpts[0, :2]
        tip = d.kpts[1, :2]
        raw_angle = pose.pointer_angle_deg(center, tip)
        angle = stabilizer.update(d.cls_id, raw_angle) if stabilizer is not None else raw_angle
        dial = counter.update(d.cls_id, raw_angle, d.score)
        if dial is not None:
            dial["raw_angle"] = float(raw_angle)
            dial["stable_angle"] = float(angle)
            dials.append(dial)
    dials.sort(key=lambda x: DIAL_LABELS.index(x["label"]) if x["label"] in DIAL_LABELS else 99)
    return dials


class VideoWorker(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.config = load_config()
        self.counter = TurnCounter(self.config, args.turn_deadband)
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.latest_dets = []
        self.latest_infer_ms = 0.0
        self.status = {
            "device": args.device, "width": args.width, "height": args.height,
            "req_fps": args.fps, "model": args.model, "model_mode": model_mode_from_path(args.model), "fps": 0.0,
            "infer_ms": 0.0, "det_count": 0, "frame_count": 0,
            "dials": [], "decimal_reading": 0.0, "cumulative_reading": 0.0,
            "decimal_m3": 0.0, "total_m3": float(self.config.get("base_m3", 0.0)),
            "cumulative_m3": 0.0, "base_m3": float(self.config.get("base_m3", 0.0)),
            "elapsed_m3": 0.0, "measurement_source": DIAL_LABELS[PRIMARY_VOLUME_DIAL],
            "measurement_source_config": self.config.get("measurement_source", "auto"),
            "error": "starting", "updated": time.time(), "core_mask": args.core_mask,
            "stream_width": args.stream_width, "stream_fps": args.stream_fps, "jpeg_quality": args.jpeg_quality,
            "paused": False, "overlay_enabled": True,
            "angle_deadband": args.angle_deadband, "angle_confirm_frames": args.angle_confirm_frames,
            "turn_deadband": args.turn_deadband,
        }
        self.stop_event = threading.Event()
        self.paused = False
        self.overlay_enabled = True
        self.stream_width = int(args.stream_width)
        self.stream_fps = float(args.stream_fps)
        self.jpeg_quality = int(args.jpeg_quality)
        self.source = None
        self.rknn = None

    def update_status(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)
            self.status["updated"] = time.time()

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg

    def set_frame(self, jpg):
        with self.lock:
            self.frame_jpeg = jpg

    def reset_turns(self):
        with self.lock:
            self.counter.reset_turns()
            self.status["cumulative_reading"] = 0.0
            self.status["cumulative_m3"] = 0.0
            self.status["elapsed_m3"] = 0.0
            self.status["total_m3"] = float(self.config.get("base_m3", 0.0))
            self.status["measurement_source"] = "自动:" + DIAL_LABELS[PRIMARY_VOLUME_DIAL] if self.config.get("measurement_source", "auto") == "auto" else DIAL_LABELS[int(self.config.get("measurement_source", PRIMARY_VOLUME_DIAL))]
            self.status["measurement_source_config"] = self.config.get("measurement_source", "auto")
        return {"ok": True, "message": "已从当前角度开始测量，本次经过水量已清零"}

    def calibrate_zero(self):
        with self.lock:
            self.counter.calibrate_zero()
            self.config = self.counter.config
            self.status["cumulative_reading"] = 0.0
            self.status["cumulative_m3"] = 0.0
            self.status["elapsed_m3"] = 0.0
            self.status["total_m3"] = float(self.config.get("base_m3", 0.0))
            self.status["measurement_source"] = "自动:" + DIAL_LABELS[PRIMARY_VOLUME_DIAL] if self.config.get("measurement_source", "auto") == "auto" else DIAL_LABELS[int(self.config.get("measurement_source", PRIMARY_VOLUME_DIAL))]
            self.status["measurement_source_config"] = self.config.get("measurement_source", "auto")
        return {"ok": True, "message": "已将当前角度设为零位，并从当前角度重新开始测量"}

    def set_measurement_source(self, source):
        source = str(source or "auto").strip().lower()
        if source not in ("auto", "0", "1", "2", "3"):
            return {"ok": False, "message": "测量来源无效"}
        with self.lock:
            self.config["measurement_source"] = source
            self.counter.config["measurement_source"] = source
            save_config(self.config)
            self.status["measurement_source_config"] = source
            if source == "auto":
                self.status["measurement_source"] = "自动:" + DIAL_LABELS[PRIMARY_VOLUME_DIAL]
                msg = "测量来源已设为自动"
            else:
                self.status["measurement_source"] = DIAL_LABELS[int(source)]
                msg = f"测量来源已设为 {DIAL_LABELS[int(source)]}"
        return {"ok": True, "message": msg}

    def set_dial_direction(self, idx, direction):
        try:
            idx = int(idx)
        except Exception:
            return {"ok": False, "message": "表盘编号无效"}
        if idx < 0 or idx >= 4:
            return {"ok": False, "message": "表盘编号超出范围"}
        direction = 1 if int(direction) >= 0 else -1
        with self.lock:
            self.config["directions"][idx] = direction
            self.counter.config["directions"][idx] = direction
            self.counter.reset_turns()
            save_config(self.config)
            self.status["elapsed_m3"] = 0.0
            self.status["cumulative_m3"] = 0.0
            self.status["cumulative_reading"] = 0.0
            self.status["total_m3"] = float(self.config.get("base_m3", 0.0))
            self.status["measurement_source_config"] = self.config.get("measurement_source", "auto")
        label = DIAL_LABELS[idx]
        name = "正向" if direction > 0 else "反向"
        return {"ok": True, "message": f"{label} 已设为{name}，本次经过水量已清零"}

    def set_base_m3(self, value):
        try:
            base_m3 = max(0.0, float(value))
        except Exception:
            return {"ok": False, "message": "m³ 基值无效"}
        with self.lock:
            self.config["base_m3"] = base_m3
            self.counter.config["base_m3"] = base_m3
            save_config(self.config)
            elapsed_m3 = float(self.status.get("elapsed_m3", 0.0))
            self.status["base_m3"] = base_m3
            self.status["total_m3"] = base_m3 + elapsed_m3
        return {"ok": True, "message": f"m³ 基值已保存：{base_m3:.3f}"}

    def set_pause(self, paused):
        paused = bool(paused)
        with self.lock:
            self.paused = paused
            self.status["paused"] = paused
            self.status["updated"] = time.time()
        return {"ok": True, "message": "视频与推理已暂停" if paused else "视频与推理已继续"}

    def set_overlay(self, enabled):
        enabled = bool(enabled)
        with self.lock:
            self.overlay_enabled = enabled
            self.status["overlay_enabled"] = enabled
            self.status["updated"] = time.time()
        return {"ok": True, "message": "检测标注已显示" if enabled else "检测标注已隐藏"}

    def set_stream(self, stream_width=None, jpeg_quality=None, stream_fps=None):
        def clamp_int(value, default, low, high):
            try:
                value = int(float(value))
            except Exception:
                value = default
            return max(low, min(high, value))

        def clamp_float(value, default, low, high):
            try:
                value = float(value)
            except Exception:
                value = default
            return max(low, min(high, value))

        with self.lock:
            self.stream_width = clamp_int(stream_width, self.stream_width, 480, int(self.args.width))
            self.jpeg_quality = clamp_int(jpeg_quality, self.jpeg_quality, 45, 98)
            self.stream_fps = clamp_float(stream_fps, self.stream_fps, 2.0, 30.0)
            self.status["stream_width"] = self.stream_width
            self.status["jpeg_quality"] = self.jpeg_quality
            self.status["stream_fps"] = self.stream_fps
            self.status["updated"] = time.time()
            msg = f"推流已更新：{self.stream_width}px / JPEG {self.jpeg_quality} / {self.stream_fps:.0f}fps"
        return {"ok": True, "message": msg}

    def restart_model_mode(self, mode):
        mode = str(mode or "").strip().lower()
        if mode not in ("accuracy", "fast"):
            return {"ok": False, "message": "未知模型模式"}
        env = os.environ.copy()
        env["WM_MODEL_MODE"] = mode
        subprocess.Popen(
            ["/home/demo/water_meter/run_hdmi_yolo11_pose_web.sh"],
            cwd="/home/demo/water_meter",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"ok": True, "message": "正在切换到精度 FP 模式" if mode == "accuracy" else "正在切换到快速 INT8 模式"}

    def run(self):
        args = self.args
        try:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(args.model)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed: {ret}")
            ret = self.rknn.init_runtime(core_mask=core_mask_value(args.core_mask))
            if ret != 0:
                raise RuntimeError(f"init_runtime failed: {ret}")
            input_w, input_h = args.input_width, args.input_height
            layout, dtype = args.input_layout, args.input_dtype
            if layout == "auto" or dtype == "auto":
                probed = det.probe_input_config(self.rknn, input_w, input_h, layout, dtype, args.model)
                if probed is None:
                    raise RuntimeError("failed to probe input config")
                input_w, input_h, layout, dtype = probed
            self.source = HDMIFrameSource(args.device, args.width, args.height, args.fps)
            self.source.open()
            stabilizer = LockedAngleStabilizer(args.angle_deadband, args.angle_alpha, args.angle_confirm_frames, args.angle_confirm_band)
            geometry = GeometryStabilizer(args.center_deadband, args.radius_deadband)
            frame_count = 0
            infer_count = 0
            t0 = time.time()
            last_loop = t0
            last_jpeg = 0.0
            fps_smooth = 0.0
            stream_interval = 1.0 / max(float(args.stream_fps), 0.1)
            self.update_status(error="", input_width=input_w, input_height=input_h, layout=layout, dtype=dtype)
            while not self.stop_event.is_set():
                with self.lock:
                    paused = self.paused
                    overlay_enabled = self.overlay_enabled
                    stream_width = self.stream_width
                    stream_fps = self.stream_fps
                    jpeg_quality = self.jpeg_quality
                if paused:
                    self.update_status(paused=True)
                    time.sleep(0.05)
                    continue
                frame = self.source.read(timeout_sec=2.0)
                if frame is None:
                    self.update_status(error="HDMI timeout")
                    continue
                frame_count += 1
                do_infer = (frame_count == 1) or (frame_count % max(1, args.infer_every) == 0)
                if do_infer:
                    dets, infer_ms = pose.infer_frame(self.rknn, frame, args, input_w, input_h, layout, dtype, debug=False)
                    self.latest_dets = keep_best_per_class(dets)
                    self.latest_infer_ms = infer_ms
                    infer_count += 1
                now = time.time()
                dt = now - last_loop
                last_loop = now
                if dt > 0:
                    fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt
                stream_interval = 1.0 / max(float(stream_fps), 0.1)
                make_jpeg = (now - last_jpeg) >= stream_interval
                dials = self.status.get("dials", [])
                if make_jpeg:
                    annotated = frame.copy()
                    dials = draw_pose_stable(annotated, self.latest_dets, stabilizer, self.counter, geometry=geometry, overlay_enabled=overlay_enabled)
                    if stream_width and annotated.shape[1] > stream_width:
                        scale = stream_width / float(annotated.shape[1])
                        annotated = cv2.resize(annotated, (stream_width, int(annotated.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                    if ok:
                        self.set_frame(enc.tobytes())
                    last_jpeg = now
                elif do_infer:
                    dials = update_dials_only(self.latest_dets, stabilizer, self.counter)
                instant_decimal_m3 = sum(float(d.get("reading_part", 0.0)) for d in dials)
                elapsed_m3, measurement_source = self.counter.elapsed_m3(dials, self.config.get("measurement_source", "auto"))
                base_m3 = float(self.config.get("base_m3", 0.0))
                total_m3 = base_m3 + elapsed_m3
                infer_fps = infer_count / max(now - t0, 1e-6)
                self.update_status(error="", fps=infer_fps, display_fps=fps_smooth, infer_ms=float(self.latest_infer_ms), det_count=len(self.latest_dets), frame_count=frame_count, dials=dials, decimal_reading=instant_decimal_m3, cumulative_reading=elapsed_m3, decimal_m3=instant_decimal_m3, total_m3=total_m3, cumulative_m3=elapsed_m3, elapsed_m3=elapsed_m3, measurement_source=measurement_source, measurement_source_config=self.config.get("measurement_source", "auto"), base_m3=base_m3, paused=False, overlay_enabled=overlay_enabled, stream_width=stream_width, stream_fps=stream_fps, jpeg_quality=jpeg_quality)
                if args.print_every and (frame_count == 1 or frame_count % args.print_every == 0):
                    print(f"[STAT] frame={frame_count} infer_fps={infer_fps:.2f} loop_fps={fps_smooth:.2f} infer={self.latest_infer_ms:.1f}ms det={len(self.latest_dets)} elapsed_m3={elapsed_m3:.6f} source={measurement_source}", flush=True)
        except Exception as exc:
            self.update_status(error=str(exc))
            print(f"[ERROR] {exc}", flush=True)
        finally:
            if self.source is not None:
                self.source.close()
            if self.rknn is not None:
                self.rknn.release()

    def stop(self):
        self.stop_event.set()


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "WaterMeterPoseWeb/2.0"

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, data, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", code)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/status":
            self.send_json(self.server.worker.get_status())
        elif path == "/snapshot.jpg":
            frame = self.server.worker.get_frame()
            if frame is None:
                self.send_error(503, "frame not ready")
            else:
                self.send_bytes(frame, "image/jpeg")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            while not self.server.worker.stop_event.is_set():
                frame = self.server.worker.get_frame()
                if frame is None or frame is last:
                    time.sleep(0.03)
                    continue
                last = frame
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body or "{}")
        except Exception:
            payload = {}
        if path != "/control":
            self.send_error(404, "not found")
            return
        action = payload.get("action", "")
        if action == "reset_turns":
            self.send_json(self.server.worker.reset_turns())
        elif action == "calibrate_zero":
            self.send_json(self.server.worker.calibrate_zero())
        elif action == "set_base_m3":
            self.send_json(self.server.worker.set_base_m3(payload.get("base_m3", 0.0)))
        elif action == "set_measurement_source":
            self.send_json(self.server.worker.set_measurement_source(payload.get("source", "auto")))
        elif action == "set_dial_direction":
            self.send_json(self.server.worker.set_dial_direction(payload.get("idx", -1), payload.get("direction", 1)))
        elif action == "set_pause":
            self.send_json(self.server.worker.set_pause(payload.get("paused", False)))
        elif action == "set_overlay":
            self.send_json(self.server.worker.set_overlay(payload.get("overlay", True)))
        elif action == "set_stream":
            self.send_json(self.server.worker.set_stream(payload.get("stream_width"), payload.get("jpeg_quality"), payload.get("stream_fps")))
        elif action == "restart_model_mode":
            self.send_json(self.server.worker.restart_model_mode(payload.get("mode", "accuracy")))
        else:
            self.send_json({"ok": False, "message": "未知操作"}, 400)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    args = parse_args()
    worker = VideoWorker(args)
    worker.start()
    server = ThreadedHTTPServer((args.host, args.port), DashboardHandler)
    server.worker = worker
    def shutdown(signum, frame):
        print("[INFO] stopping web dashboard", flush=True)
        worker.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"[INFO] web dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        worker.stop()
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
