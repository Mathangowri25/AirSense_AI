import streamlit as st
import pickle, numpy as np, os, base64, math, io
import pandas as pd
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()  # Only works locally — ignored in Streamlit Cloud

# Works both locally (.env) and in Streamlit Cloud (Secrets Manager)
GROQ_API_KEY_LLM    = st.secrets.get("GROK_API")    or os.getenv("GROK_API")
GROQ_API_KEY_SPEECH = st.secrets.get("GROK_SPEECH") or os.getenv("GROK_SPEECH")

st.set_page_config(page_title="AQI Intelligence", page_icon="🌐",
                   layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Exo+2:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp {
    background: #050d0a !important;
    background-image:
        radial-gradient(ellipse 70% 55% at 15% 25%, rgba(14,90,65,0.22) 0%, transparent 65%),
        radial-gradient(ellipse 55% 45% at 85% 75%, rgba(20,80,100,0.18) 0%, transparent 65%),
        radial-gradient(ellipse 45% 60% at 55% 5%,  rgba(40,60,20,0.14) 0%, transparent 60%) !important;
    animation: env-breathe 18s ease-in-out infinite alternate !important;
}
@keyframes env-breathe {
    0%   { background-position: 0% 0%, 100% 100%, 55% 5%; }
    100% { background-position: 8% 12%, 92% 88%, 62% 15%; }
}
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(20,184,154,0.008) 3px, rgba(20,184,154,0.008) 4px);
}
.main, .block-container {
    background: transparent !important;
    padding-top: 1rem !important;
    max-width: 100% !important;
}
[data-testid="collapsedControl"] { display:none !important; }
section[data-testid="stSidebar"]  { display:none !important; }

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: rgba(10,26,20,0.7) !important;
    border: 1px dashed rgba(20,184,154,0.4) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px) !important;
}
[data-testid="stFileUploader"] * { color: #14b89a !important; font-size: 13px !important; }
[data-testid="stFileUploader"] button {
    background: rgba(20,184,154,0.10) !important;
    border: 1px solid rgba(20,184,154,0.5) !important;
    color: #14b89a !important; border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(5,18,12,0.9) !important;
    border: 1px solid rgba(20,184,154,0.35) !important;
    border-radius: 8px !important; color: #14b89a !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
}
.stButton > button {
    background: rgba(20,184,154,0.08) !important;
    color: #14b89a !important;
    border: 1.5px solid rgba(20,184,154,0.7) !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 13px !important; letter-spacing: .12em !important;
    text-transform: uppercase !important; padding: 14px 32px !important;
    transition: all .22s !important;
    box-shadow: 0 0 20px rgba(20,184,154,0.12) !important;
}
.stButton > button:hover {
    background: rgba(20,184,154,0.14) !important;
    box-shadow: 0 0 32px rgba(20,184,154,0.28) !important;
    border-color: #14b89a !important;
}
.stSpinner > div { border-top-color: #14b89a !important; }

/* ══════════════════════════════════════════════════════
   ANIMATED PAGE HEADER
══════════════════════════════════════════════════════ */
.page-header-wrap {
    margin-bottom: 28px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(20,184,154,.15);
    position: relative;
    overflow: hidden;
}
.page-header-glow {
    position: absolute; top: -60px; left: -80px;
    width: 420px; height: 200px;
    background: radial-gradient(ellipse, rgba(20,184,154,.18) 0%, transparent 70%);
    animation: hdr-glow-drift 6s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes hdr-glow-drift {
    0%   { transform: translate(0px,  0px)  scale(1.0); opacity:.7; }
    100% { transform: translate(60px, 20px) scale(1.2); opacity:1; }
}
.page-title {
    font-family: 'Orbitron', monospace;
    font-size: 30px; font-weight: 900;
    background: linear-gradient(90deg, #14b89a 0%, #4ab3d4 40%, #8a6fe8 80%, #14b89a 100%);
    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: title-shimmer 5s linear infinite;
    letter-spacing: .07em;
    margin-bottom: 8px;
    display: inline-block;
}
@keyframes title-shimmer {
    0%   { background-position: 0%   center; }
    100% { background-position: 250% center; }
}
.page-title-underline {
    height: 3px; width: 0px;
    background: linear-gradient(90deg, #14b89a, #4ab3d4, #8a6fe8);
    border-radius: 2px;
    animation: underline-grow 1.4s cubic-bezier(.4,0,.2,1) forwards .3s;
    margin-bottom: 12px;
}
@keyframes underline-grow { to { width: 520px; } }
.page-tags { display: flex; gap: 14px; flex-wrap: wrap; }
.page-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: .14em;
    padding: 3px 12px; border-radius: 20px;
    border: 1px solid rgba(20,184,154,.25);
    color: rgba(20,184,154,.6);
    animation: tag-fade-in .5s ease forwards;
    opacity: 0;
}
.page-tag:nth-child(1){ animation-delay:.5s; }
.page-tag:nth-child(2){ animation-delay:.7s; }
.page-tag:nth-child(3){ animation-delay:.9s; }
.page-tag:nth-child(4){ animation-delay:1.1s; }
.page-tag:nth-child(5){ animation-delay:1.3s; }
@keyframes tag-fade-in { to { opacity:1; } }

/* ══════════════════════════════════════════════════════
   SECTION HEADERS — bold colored
══════════════════════════════════════════════════════ */
.sec-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 14px;
    position: relative;
}
.sec-header::after {
    content: '';
    position: absolute; bottom: 0; left: 0;
    height: 1px; width: 100%;
    background: linear-gradient(90deg, var(--sec-clr) 0%, transparent 70%);
    opacity: .28;
}
.sec-num {
    font-family: 'Orbitron', monospace;
    font-size: 28px; font-weight: 900;
    color: var(--sec-clr);
    opacity: .18; letter-spacing: -.02em;
    min-width: 44px;
}
.sec-bar {
    width: 4px; min-height: 42px; border-radius: 2px;
    background: linear-gradient(180deg, var(--sec-clr), transparent);
    box-shadow: 0 0 10px var(--sec-clr);
    flex-shrink: 0;
}
.sec-text-wrap { display: flex; flex-direction: column; gap: 2px; }
.sec-icon {
    font-family: 'Orbitron', monospace;
    font-size: 11px; color: var(--sec-clr);
    opacity: .5; letter-spacing: .14em;
}
.sec-title {
    font-family: 'Orbitron', monospace;
    font-size: 18px; font-weight: 700;
    color: var(--sec-clr);
    letter-spacing: .1em; text-transform: uppercase;
    text-shadow: 0 0 18px var(--sec-clr);
    animation: sec-pulse 3.5s ease-in-out infinite;
}
@keyframes sec-pulse {
    0%,100% { text-shadow: 0 0 14px var(--sec-clr); }
    50%      { text-shadow: 0 0 28px var(--sec-clr), 0 0 50px var(--sec-clr); }
}
.sec-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--sec-clr); opacity: .45;
    letter-spacing: .12em;
}

/* ══════════════════════════════════════════════════════
   AI ANALYSIS ANIMATED BOX
══════════════════════════════════════════════════════ */
.ai-terminal {
    background: rgba(3,12,8,.9);
    border: 1px solid rgba(20,184,154,.18);
    border-radius: 12px;
    padding: 22px 26px;
    position: relative;
    overflow: hidden;
}
.ai-terminal::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 60%; height: 2px;
    background: linear-gradient(90deg, transparent, #14b89a, transparent);
    animation: scan-line 4s linear infinite;
}
@keyframes scan-line { to { left: 150%; } }
.ai-terminal-bar {
    display: flex; align-items: center; gap: 7px;
    margin-bottom: 18px; padding-bottom: 12px;
    border-bottom: 1px solid rgba(20,184,154,.1);
}
.ai-dot { width: 11px; height: 11px; border-radius: 50%; }
.ai-filename {
    margin-left: 10px; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: rgba(20,184,154,.4); letter-spacing: .1em;
}
.ai-blink-cursor {
    display: inline-block;
    width: 9px; height: 16px;
    background: #14b89a;
    margin-left: 3px; vertical-align: middle;
    animation: cursor-blink .85s step-end infinite;
    border-radius: 1px;
}
@keyframes cursor-blink { 0%,100%{opacity:1;} 50%{opacity:0;} }
.ai-line {
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.9; margin-bottom: 3px;
    overflow: hidden; white-space: normal;
    animation: line-reveal .4s ease forwards;
    opacity: 0;
}
@keyframes line-reveal { to { opacity: 1; } }
.ai-line-header {
    font-size: 15px; font-weight: 700;
    color: #14b89a;
    text-shadow: 0 0 12px rgba(20,184,154,.55);
    padding: 6px 0 2px 0;
}
.ai-line-body {
    font-size: 13px;
    color: rgba(130,210,170,.82);
    padding-left: 14px;
}

/* Metric + misc */
.neon-card {
    background: rgba(5,16,11,.75); border-radius: 10px;
    padding: 15px 10px; text-align: center;
    position: relative; overflow: hidden;
}
.neon-card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--card-clr),transparent);
}
</style>

<canvas id="starfield" style="position:fixed;top:0;left:0;width:100%;height:100%;
    pointer-events:none;z-index:0;opacity:0.5;"></canvas>
<script>
(function(){
const cv=document.getElementById('starfield');
if(!cv)return;
const ctx=cv.getContext('2d');
function resize(){cv.width=window.innerWidth;cv.height=window.innerHeight;}
resize();window.addEventListener('resize',resize);
const stars=Array.from({length:140},()=>({
    x:Math.random()*cv.width,y:Math.random()*cv.height,
    r:Math.random()*1.6+.3,a:Math.random(),
    s:Math.random()*.003+.001,
    c:Math.random()<.6?'20,184,154':Math.random()<.5?'74,179,212':'138,111,232'
}));
function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    stars.forEach(s=>{
        s.a+=s.s;if(s.a>1||s.a<0)s.s*=-1;
        ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
        ctx.fillStyle=`rgba(${s.c},${s.a})`;
        ctx.shadowColor=`rgba(${s.c},0.7)`;ctx.shadowBlur=5;
        ctx.fill();});
    requestAnimationFrame(draw);}
draw();
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_3d_viz(aqi_val, category):
    cmap = {
        "Good":("0x14b89a","0x0d8a72","#14b89a"),
        "Moderate":("0xe8a443","0xb07830","#e8a443"),
        "Unhealthy for Sensitive Groups":("0xe8723a","0xb05228","#e8723a"),
        "Unhealthy":("0xe85a3a","0xb03d24","#e85a3a"),
        "Very Unhealthy":("0x8a6fe8","0x6248c0","#8a6fe8"),
        "Hazardous":("0xd43a2a","0x9c2218","#d43a2a"),
    }
    c1,c2,css=cmap.get(category,("0x14b89a","0x0d8a72","#14b89a"))
    pc=min(int(aqi_val*2.0),600); spd=0.003+(aqi_val/500)*0.016; chaos=0.4+(aqi_val/500)*3.0
    return f"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0;}}body{{background:transparent;overflow:hidden;}}canvas{{display:block;}}
#ov{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;}}
#n{{font-family:'Orbitron',monospace;font-size:68px;font-weight:900;color:{css};
    text-shadow:0 0 24px {css},0 0 50px {css}88,0 0 90px {css}33;line-height:1;}}
#l{{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.22em;color:{css}cc;
    text-transform:uppercase;margin-top:7px;}}
#c{{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.12em;color:{css}aa;
    text-transform:uppercase;margin-top:4px;}}
</style></head><body>
<div id="ov"><div id="n">{aqi_val}</div><div id="l">AQI INDEX</div><div id="c">{category}</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W=window.innerWidth||500,H=window.innerHeight||380;
const renderer=new THREE.WebGLRenderer({{antialias:true,alpha:true}});
renderer.setSize(W,H);renderer.setPixelRatio(devicePixelRatio);
renderer.setClearColor(0,0);document.body.appendChild(renderer.domElement);
const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(56,W/H,0.1,1000);
cam.position.z=3.2;const clk=new THREE.Clock();
const cM=new THREE.MeshPhongMaterial({{color:{c1},emissive:{c1},emissiveIntensity:.5,transparent:true,opacity:.20}});
const core=new THREE.Mesh(new THREE.SphereGeometry(.52,64,64),cM);scene.add(core);
const wM=new THREE.MeshBasicMaterial({{color:{c1},wireframe:true,transparent:true,opacity:.18}});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(.56,20,20),wM));
const iM=new THREE.MeshBasicMaterial({{color:{c2},wireframe:true,transparent:true,opacity:.25}});
const ico=new THREE.Mesh(new THREE.IcosahedronGeometry(.72,1),iM);scene.add(ico);
const mkR=(r,op,tilt)=>{{const m=new THREE.Mesh(new THREE.TorusGeometry(r,.007,8,120),
    new THREE.MeshBasicMaterial({{color:{c1},transparent:true,opacity:op}}));m.rotation.x=tilt;return m;}};
const r1=mkR(.88,.75,Math.PI/2);scene.add(r1);
const r2=mkR(1.05,.45,Math.PI/3);r2.rotation.z=1;scene.add(r2);
const r3=mkR(1.22,.28,Math.PI/5);r3.rotation.z=2.5;scene.add(r3);
const COUNT={pc};const pos=new Float32Array(COUNT*3);const vel=[];
for(let i=0;i<COUNT;i++){{const th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1);
    const r=1.3*(0.7+Math.random()*.6);
    pos[i*3]=r*Math.sin(ph)*Math.cos(th);pos[i*3+1]=r*Math.sin(ph)*Math.sin(th);pos[i*3+2]=r*Math.cos(ph);
    vel.push({{vt:(Math.random()-.5)*{chaos}*.01,vp:(Math.random()-.5)*{chaos}*.01,
               r,t:Math.random()*Math.PI*2,s:.002+Math.random()*.006}});}}
const pG=new THREE.BufferGeometry();pG.setAttribute('position',new THREE.BufferAttribute(pos,3));
const pts=new THREE.Points(pG,new THREE.PointsMaterial({{color:{c1},size:.027,transparent:true,opacity:.88}}));
scene.add(pts);
const ng=new THREE.Group();const np=[];
for(let i=0;i<12;i++){{const m=new THREE.Mesh(new THREE.SphereGeometry(.04,12,12),
    new THREE.MeshPhongMaterial({{color:i%2?{c2}:{c1},emissive:i%2?{c2}:{c1},emissiveIntensity:1.0}}));
    const a=(i/12)*Math.PI*2,rr=.65+(i%3)*.12;
    m.position.set(rr*Math.cos(a),(Math.random()-.5)*.5,rr*Math.sin(a));np.push(m.position.clone());ng.add(m);}}
for(let i=0;i<12;i++){{const g=new THREE.BufferGeometry().setFromPoints([np[i],np[(i+1)%12]]);
    ng.add(new THREE.Line(g,new THREE.LineBasicMaterial({{color:{c1},transparent:true,opacity:.38}})));}}
scene.add(ng);
scene.add(new THREE.AmbientLight(0xffffff,.3));
const pl=new THREE.PointLight({c1},3.5,10);pl.position.set(2,2,2);scene.add(pl);
const pl2=new THREE.PointLight({c2},2.2,10);pl2.position.set(-2,-1,-1);scene.add(pl2);
const SP={spd};let fr=0;
function animate(){{requestAnimationFrame(animate);const t=clk.getElapsedTime();fr++;
    core.rotation.y+=SP*.7;core.rotation.x+=SP*.3;
    ico.rotation.y-=SP*.4;ico.rotation.z+=SP*.22;
    r1.rotation.z+=SP*.5;r2.rotation.y+=SP*.65;r3.rotation.x+=SP*.28;
    pts.rotation.y+=SP;pts.rotation.x+=SP*.25;
    ng.rotation.y+=SP*1.4;ng.rotation.x=Math.sin(t*.35)*.20;
    cM.emissiveIntensity=.42+Math.sin(t*1.6)*.20;cM.opacity=.16+Math.sin(t*1.2)*.08;
    if(fr%2===0){{for(let i=0;i<COUNT;i++){{const v=vel[i];v.t+=v.s;
        pos[i*3]+=v.vt;pos[i*3+1]+=v.vp;pos[i*3+2]+=Math.sin(v.t)*.002;
        const len=Math.sqrt(pos[i*3]**2+pos[i*3+1]**2+pos[i*3+2]**2);
        if(len>0){{const r=v.r*(1+.035*Math.sin(v.t));
            pos[i*3]=(pos[i*3]/len)*r;pos[i*3+1]=(pos[i*3+1]/len)*r;pos[i*3+2]=(pos[i*3+2]/len)*r;}}}}
        pG.attributes.position.needsUpdate=true;}}
    renderer.render(scene,cam);}}
animate();
</script></body></html>"""

def get_aqi_gauge(aqi_val, category):
    cat_css={"Good":"#14b89a","Moderate":"#e8a443","Unhealthy for Sensitive Groups":"#e8723a",
             "Unhealthy":"#e85a3a","Very Unhealthy":"#8a6fe8","Hazardous":"#d43a2a"}.get(category,"#14b89a")
    return f"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0;}}body{{background:transparent;overflow:hidden;font-family:'JetBrains Mono',monospace;}}
#wrap{{display:flex;flex-direction:column;align-items:center;padding:8px 0;}}
#title{{font-family:'Orbitron',monospace;font-size:11px;letter-spacing:.20em;
        color:rgba(20,184,154,0.6);text-transform:uppercase;margin-bottom:8px;}}
#cl{{font-family:'Orbitron',monospace;font-size:13px;letter-spacing:.13em;color:{cat_css};
     text-transform:uppercase;margin-top:6px;text-shadow:0 0 14px {cat_css}88;}}
#leg{{display:flex;gap:7px;flex-wrap:wrap;justify-content:center;margin-top:12px;padding:0 8px;}}
.li{{display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(20,184,154,0.65);}}
.ld{{width:8px;height:8px;border-radius:50%;flex-shrink:0;}}
</style></head><body>
<div id="wrap">
<div id="title">◈ AQI LEVEL SCALE</div>
<canvas id="c" width="440" height="210"></canvas>
<div id="cl">▲ {category} · AQI {aqi_val}</div>
<div id="leg">
  <div class="li"><div class="ld" style="background:#14b89a"></div>0–50</div>
  <div class="li"><div class="ld" style="background:#e8a443"></div>51–100</div>
  <div class="li"><div class="ld" style="background:#e8723a"></div>101–150</div>
  <div class="li"><div class="ld" style="background:#e85a3a"></div>151–200</div>
  <div class="li"><div class="ld" style="background:#8a6fe8"></div>201–300</div>
  <div class="li"><div class="ld" style="background:#d43a2a"></div>301+</div>
</div></div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=cv.width,H=cv.height,cx=W/2,cy=H-22,R=145,Ri=95;
const zones=[{{min:0,max:50,color:'#14b89a'}},{{min:50,max:100,color:'#e8a443'}},
  {{min:100,max:150,color:'#e8723a'}},{{min:150,max:200,color:'#e85a3a'}},
  {{min:200,max:300,color:'#8a6fe8'}},{{min:300,max:500,color:'#d43a2a'}}];
function a2r(v){{return Math.PI-(Math.min(v,500)/500)*Math.PI;}}
function drawBase(){{
    ctx.clearRect(0,0,W,H);
    ctx.beginPath();ctx.arc(cx,cy,R,Math.PI,0,false);ctx.arc(cx,cy,Ri,0,Math.PI,true);
    ctx.closePath();ctx.fillStyle='rgba(5,20,14,.7)';ctx.fill();
    zones.forEach(z=>{{const a1=a2r(z.min),a2=a2r(z.max);
        ctx.beginPath();ctx.arc(cx,cy,R,a1,a2,true);ctx.arc(cx,cy,Ri,a2,a1,false);
        ctx.closePath();ctx.fillStyle=z.color+'bb';ctx.fill();
        ctx.strokeStyle='#030e07';ctx.lineWidth=1.5;ctx.stroke();}});
    [0,50,100,150,200,300,500].forEach(v=>{{
        const a=a2r(v);
        ctx.beginPath();ctx.moveTo(cx+(Ri-5)*Math.cos(a),cy-(Ri-5)*Math.sin(a));
        ctx.lineTo(cx+(R+5)*Math.cos(a),cy-(R+5)*Math.sin(a));
        ctx.strokeStyle='rgba(20,184,154,0.7)';ctx.lineWidth=1.5;ctx.stroke();
        ctx.fillStyle='rgba(20,184,154,0.75)';ctx.font='bold 9px JetBrains Mono,monospace';
        ctx.textAlign='center';ctx.textBaseline='middle';
        ctx.fillText(String(v),cx+(R+16)*Math.cos(a),cy-(R+16)*Math.sin(a));}});}}
const T={aqi_val},TA=a2r(T),CC='{cat_css}';
let cur=Math.PI,fr=0,pt=0;
function drawNeedle(angle){{
    const nl=R-10,nx=cx+nl*Math.cos(angle),ny=cy-nl*Math.sin(angle);
    const g=ctx.createLinearGradient(cx,cy,nx,ny);
    g.addColorStop(0,CC+'00');g.addColorStop(1,CC+'ff');
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);
    ctx.strokeStyle=g;ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);
    ctx.strokeStyle=CC;ctx.lineWidth=2.5;ctx.stroke();
    ctx.shadowColor=CC;ctx.shadowBlur=16;
    ctx.beginPath();ctx.arc(nx,ny,6,0,Math.PI*2);ctx.fillStyle=CC;ctx.fill();
    ctx.shadowBlur=0;
    const pulse=1+.10*Math.sin(pt*3);
    const hg=ctx.createRadialGradient(cx,cy,0,cx,cy,15*pulse);
    hg.addColorStop(0,CC+'aa');hg.addColorStop(1,CC+'00');
    ctx.beginPath();ctx.arc(cx,cy,15*pulse,0,Math.PI*2);ctx.fillStyle=hg;ctx.fill();
    ctx.shadowColor=CC;ctx.shadowBlur=22;
    ctx.fillStyle=CC;ctx.font='bold 32px Orbitron,monospace';
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(T),cx,cy-40);
    ctx.shadowBlur=12;ctx.fillStyle=CC+'bb';ctx.font='bold 10px JetBrains Mono,monospace';
    ctx.fillText('{category}'.toUpperCase().slice(0,22),cx,cy-22);ctx.shadowBlur=0;}}
function animate(){{requestAnimationFrame(animate);pt+=.016;
    if(fr<100){{const t=fr/100,e=t<.5?2*t*t:-1+(4-2*t)*t;cur=Math.PI+(TA-Math.PI)*e;fr++;}}
    else cur=TA;drawBase();drawNeedle(cur);}}
animate();
</script></body></html>"""

def get_radar(pm25,pm10,no2,so2,o3,co,co2,traf):
    vn=[round(min(pm25/300,1)*100,1),round(min(pm10/400,1)*100,1),round(min(no2/200,1)*100,1),
        round(min(so2/200,1)*100,1),round(min(o3/200,1)*100,1),round(min(co/30,1)*100,1),
        round(min(co2/2000,1)*100,1),round(min(traf/100,1)*100,1)]
    return rf"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0;}}body{{background:transparent;overflow:hidden;
    display:flex;align-items:center;justify-content:center;}}canvas{{display:block;}}</style></head><body>
<canvas id="c" width="400" height="400"></canvas>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=cv.width,H=cv.height,cx=W/2,cy=H/2,R=138;
const labels=['PM2.5','PM10','NO₂','SO₂','O₃','CO','CO₂','Traffic'];
const vals={vn};const N=labels.length;
const colors=['#14b89a','#e8a443','#e8723a','#e85a3a','#4ab3d4','#8a6fe8','#a3d46a','#e8c43a'];
function toXY(i,r){{const a=(i/N)*Math.PI*2-Math.PI/2;return[cx+r*Math.cos(a),cy+r*Math.sin(a)];}}
ctx.clearRect(0,0,W,H);
for(let ring=1;ring<=5;ring++){{const rr=R*(ring/5);
    ctx.beginPath();for(let i=0;i<N;i++){{const[x,y]=toXY(i,rr);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}}
    ctx.closePath();ctx.strokeStyle=`rgba(20,184,154,${{0.10+ring*.04}})`;ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle=`rgba(10,30,20,${{0.10+ring*.04}})`;ctx.fill();}}
for(let i=0;i<N;i++){{const[x,y]=toXY(i,R);
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);
    ctx.strokeStyle='rgba(20,184,154,.25)';ctx.lineWidth=1;ctx.stroke();}}
ctx.beginPath();vals.forEach((v,i)=>{{const r=R*(v/100);const[x,y]=toXY(i,r);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});ctx.closePath();
const grd=ctx.createRadialGradient(cx,cy,0,cx,cy,R);
grd.addColorStop(0,'rgba(20,184,154,.28)');grd.addColorStop(1,'rgba(232,90,58,.18)');
ctx.fillStyle=grd;ctx.fill();
ctx.shadowColor='rgba(20,184,154,.55)';ctx.shadowBlur=8;
ctx.strokeStyle='rgba(20,184,154,.80)';ctx.lineWidth=2;ctx.stroke();ctx.shadowBlur=0;
vals.forEach((v,i)=>{{const r=R*(v/100);const[x,y]=toXY(i,r);
    ctx.shadowColor=colors[i];ctx.shadowBlur=14;
    const gc=ctx.createRadialGradient(x,y,0,x,y,9);
    gc.addColorStop(0,colors[i]);gc.addColorStop(1,colors[i]+'00');
    ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.fillStyle=gc;ctx.fill();
    ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fillStyle=colors[i];ctx.fill();
    ctx.shadowBlur=0;}});
labels.forEach((lb,i)=>{{const[x,y]=toXY(i,R+22);ctx.fillStyle=colors[i];
    ctx.font='bold 12px JetBrains Mono,monospace';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.shadowColor=colors[i];ctx.shadowBlur=10;ctx.fillText(lb,x,y);ctx.shadowBlur=0;}});
const cg=ctx.createRadialGradient(cx,cy,0,cx,cy,18);
cg.addColorStop(0,'rgba(20,184,154,.5)');cg.addColorStop(1,'rgba(20,184,154,0)');
ctx.beginPath();ctx.arc(cx,cy,18,0,Math.PI*2);ctx.fillStyle=cg;ctx.fill();
ctx.beginPath();ctx.arc(cx,cy,4.5,0,Math.PI*2);ctx.fillStyle='#14b89a';ctx.fill();
</script></body></html>"""

def get_waveform(color="#8a6fe8"):
    return f"""<!DOCTYPE html><html><head>
<style>*{{margin:0;padding:0;}}body{{background:transparent;overflow:hidden;}}</style></head><body>
<canvas id="c" width="500" height="72"></canvas>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=cv.width,H=cv.height;let t=0;
function draw(){{ctx.clearRect(0,0,W,H);const bars=75;
    for(let i=0;i<bars;i++){{const x=(i/bars)*W;
        const h=(Math.sin(i*.25+t)*.5+.5)*(Math.sin(i*.1+t*.7)*.5+.5)*28+5;
        const a=.50+.42*(h/33);
        const g=ctx.createLinearGradient(0,H/2-h,0,H/2+h);
        g.addColorStop(0,'{color}'+'00');
        g.addColorStop(.5,'{color}'+Math.floor(a*255).toString(16).padStart(2,'0'));
        g.addColorStop(1,'{color}'+'00');
        ctx.shadowColor='{color}';ctx.shadowBlur=6;
        ctx.fillStyle=g;ctx.fillRect(x-1.6,H/2-h,3.2,h*2);ctx.shadowBlur=0;}}
    t+=.055;requestAnimationFrame(draw);}}
draw();
</script></body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED AI ANALYSIS BOX  — typewriter effect with JS
# ══════════════════════════════════════════════════════════════════════════════
def get_ai_analysis_html(explanation_text):
    lines = [l.strip() for l in explanation_text.replace("\n\n", "\n").split("\n") if l.strip()]
    # Build JSON-safe lines list
    import json
    lines_json = json.dumps(lines)
    return f"""<!DOCTYPE html><html><head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=JetBrains+Mono:wght@400;600&display=swap');
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:rgba(3,12,8,.92);font-family:'JetBrains Mono',monospace;
      overflow-y:auto;padding:20px 24px;}}
#terminal{{position:relative;}}
#terminal::before{{
    content:'';position:absolute;top:0;left:-24px;right:-24px;height:2px;
    background:linear-gradient(90deg,transparent,#14b89a,transparent);
    animation:scan 4s linear infinite;
}}
@keyframes scan{{from{{transform:translateX(-100%);}}to{{transform:translateX(200%);}}}}
.topbar{{display:flex;align-items:center;gap:7px;margin-bottom:16px;
          padding-bottom:12px;border-bottom:1px solid rgba(20,184,154,.12);}}
.dot{{width:11px;height:11px;border-radius:50%;display:inline-block;}}
.fname{{margin-left:10px;font-size:11px;color:rgba(20,184,154,.45);letter-spacing:.1em;}}
.status{{margin-left:auto;font-size:10px;color:rgba(20,184,154,.4);
          animation:blink-status 2s step-end infinite;}}
@keyframes blink-status{{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}
.line{{overflow:hidden;max-height:0;opacity:0;transition:max-height .3s ease,opacity .25s ease;}}
.line.visible{{max-height:200px;opacity:1;}}
.line-header{{
    font-family:'Orbitron',monospace;
    font-size:15px;font-weight:700;
    padding:8px 0 3px;
    letter-spacing:.08em;
    display:flex;align-items:center;gap:8px;
}}
.line-header::before{{
    content:'▶';font-size:10px;
    animation:chevron-pulse 1.8s ease-in-out infinite;
}}
@keyframes chevron-pulse{{0%,100%{{opacity:.5;}}50%{{opacity:1;}}}}
.line-body{{
    font-size:13px;
    color:rgba(140,215,175,.82);
    line-height:1.9;
    padding-left:16px;
    padding-bottom:2px;
    border-left:2px solid rgba(20,184,154,.18);
    margin-left:6px;
    margin-bottom:4px;
}}
.cursor{{display:inline-block;width:8px;height:14px;background:#14b89a;
          vertical-align:middle;margin-left:3px;border-radius:1px;
          animation:blink .85s step-end infinite;}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0;}}}}
/* scrollbar */
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:rgba(20,184,154,.3);border-radius:2px;}}
</style>
</head><body>
<div id="terminal">
  <div class="topbar">
    <span class="dot" style="background:#e85a3a;"></span>
    <span class="dot" style="background:#e8a443;"></span>
    <span class="dot" style="background:#14b89a;"></span>
    <span class="fname">aqi_analysis.log</span>
    <span class="status">● LIVE</span>
  </div>
  <div id="content"></div>
</div>
<script>
const lines = {lines_json};
const content = document.getElementById('content');
let idx = 0;

function isHeader(line) {{
    return /^(\*\*|#{1,3}|\d+\.|[A-Z][A-Z\s]{{6,}}:)/.test(line.trim());
}}

function headerColors(i) {{
    const palette = ['#14b89a','#4ab3d4','#e8a443','#8a6fe8','#e8723a','#a3d46a','#e8c43a'];
    return palette[i % palette.length];
}}

let hdrCount = 0;

function addLine() {{
    if (idx >= lines.length) {{
        // add blinking cursor at end
        const cur = document.createElement('span');
        cur.className = 'cursor';
        content.appendChild(cur);
        return;
    }}
    const text = lines[idx].replace(/\\*\\*/g,'').replace(/^#+\\s*/,'').trim();
    const div = document.createElement('div');
    div.className = 'line';

    if (isHeader(lines[idx])) {{
        div.className += ' line-header';
        div.style.color = headerColors(hdrCount++);
        div.style.textShadow = `0 0 14px ${{div.style.color}}88`;
        div.textContent = text;
    }} else {{
        div.className += ' line-body';
        div.textContent = text;
    }}

    content.appendChild(div);

    requestAnimationFrame(() => {{
        requestAnimationFrame(() => {{ div.classList.add('visible'); }});
    }});

    idx++;
    const delay = isHeader(lines[idx-1]) ? 80 : 45;
    setTimeout(addLine, delay);
}}

setTimeout(addLine, 300);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def classify_aqi(v):
    if 0<=v<=50:      return "Good"
    elif 51<=v<=100:  return "Moderate"
    elif 101<=v<=150: return "Unhealthy for Sensitive Groups"
    elif 151<=v<=200: return "Unhealthy"
    elif 201<=v<=300: return "Very Unhealthy"
    elif v>=301:      return "Hazardous"
    return "Invalid"

def aqi_color(cat):
    return {"Good":"#14b89a","Moderate":"#e8a443","Unhealthy for Sensitive Groups":"#e8723a",
            "Unhealthy":"#e85a3a","Very Unhealthy":"#8a6fe8","Hazardous":"#d43a2a"}.get(cat,"#14b89a")

def aqi_badge(cat):
    return {"Good":"background:rgba(20,184,154,.16);color:#14b89a;border:1.5px solid #14b89a;",
            "Moderate":"background:rgba(232,164,67,.16);color:#e8a443;border:1.5px solid #e8a443;",
            "Unhealthy for Sensitive Groups":"background:rgba(232,114,58,.16);color:#e8723a;border:1.5px solid #e8723a;",
            "Unhealthy":"background:rgba(232,90,58,.16);color:#e85a3a;border:1.5px solid #e85a3a;",
            "Very Unhealthy":"background:rgba(138,111,232,.16);color:#8a6fe8;border:1.5px solid #8a6fe8;",
            "Hazardous":"background:rgba(212,58,42,.16);color:#d43a2a;border:1.5px solid #d43a2a;",
            }.get(cat,"background:rgba(20,184,154,.12);color:#14b89a;border:1.5px solid #14b89a;")

def bat_color(b): return "#14b89a" if b>=60 else "#e8a443" if b>=30 else "#e85a3a"
def dqi_color(d): return "#14b89a" if d>=80 else "#e8a443" if d>=50 else "#e85a3a"

@st.cache_resource
def load_model():
    if os.path.exists("predict_AQI_score.pkl"):
        with open("predict_AQI_score.pkl","rb") as f: return pickle.load(f)
    return None

def predict_aqi(model, feats):
    try: return round(float(model.predict(np.array([feats]))[0]),1)
    except:
        try: return round(float(model.predict(np.array([[feats[0],feats[1],feats[4]]]))[0]),1)
        except: return round(feats[0]*1.5+feats[1]*.5+feats[4]*.3,1)

def gen_explanation(aqi,pm25,pm10,co,co2,no2,so2,o3,temp,hum,wind):
    p=f"""Expert AQI analysis. AQI={aqi}.
Pollutants: PM2.5={pm25}µg/m³, PM10={pm10}µg/m³, CO={co}ppm, CO2={co2}ppm,
NO2={no2}µg/m³, SO2={so2}µg/m³, O3={o3}µg/m³.
Environment: Temp={temp}°C, Humidity={hum}%, Wind={wind}m/s.
Provide clearly labeled sections: 1. Main Causes  2. Health Effects  3. Safety Precautions  4. Pollution Sources  5. Improvement Suggestions. Use plain text, no markdown symbols."""
    r=Groq(api_key=GROQ_API_KEY_LLM).chat.completions.create(
        model="llama-3.1-8b-instant",messages=[{"role":"user","content":p}])
    return r.choices[0].message.content

def gen_voice(expl):
    p=f"Convert to 2-3 sentence spoken air quality alert with warning tone and health advice:\n{expl}\nVoice Alert:"
    r=Groq(api_key=GROQ_API_KEY_SPEECH).chat.completions.create(
        model="llama-3.1-8b-instant",messages=[{"role":"user","content":p}])
    return r.choices[0].message.content

def to_mp3(text,path="alert.mp3"):
    gTTS(text).save(path); return path

def play_audio(path):
    with open(path,"rb") as f: b64=base64.b64encode(f.read()).decode()
    st.markdown(f'<audio controls autoplay style="width:100%;border-radius:6px;margin-top:10px;">'
                f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True)

def neon_metric(label, value, unit, color):
    return f"""<div style="--card-clr:{color};background:rgba(5,16,11,.75);border:1px solid {color}33;
        border-radius:10px;padding:15px 10px;text-align:center;position:relative;overflow:hidden;
        box-shadow:0 0 20px {color}0e;">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,transparent,{color}aa,transparent);"></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.13em;
            color:{color}66;text-transform:uppercase;margin-bottom:7px;">{label}</div>
        <div style="font-family:'Orbitron',monospace;font-size:26px;font-weight:700;
            color:{color};text-shadow:0 0 14px {color}88;line-height:1;">{value}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
            color:{color}44;margin-top:4px;">{unit}</div></div>"""

def glowing_bar(name,value,max_val,color,unit):
    pct=min(int(value/max_val*100),100)
    return f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:13px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
            color:{color}bb;width:52px;flex-shrink:0;">{name}</div>
        <div style="flex:1;height:8px;background:rgba(5,18,12,.8);border-radius:4px;
            overflow:hidden;border:1px solid {color}28;">
            <div style="width:{pct}%;height:100%;border-radius:4px;
                background:linear-gradient(90deg,{color}66,{color});
                box-shadow:0 0 10px {color}99;"></div></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
            color:{color};width:82px;text-align:right;flex-shrink:0;">
            {value} <span style="color:{color}44;font-size:10px;">{unit}</span></div></div>"""

def sec_wrap_open(accent, number, icon, title, subtitle=""):
    # Each section header: big number ghost + colored bar + large bold title
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(5,16,11,.92) 0%,rgba(3,10,7,.97) 100%);
                border:1px solid {accent}1a;border-top:2px solid {accent}55;
                border-radius:14px;padding:22px;margin-bottom:14px;
                backdrop-filter:blur(6px);
                box-shadow:0 6px 32px rgba(0,0,0,.55),0 0 40px {accent}06;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;
                  padding-bottom:14px;border-bottom:1px solid {accent}20;
                  position:relative;">
        <!-- ghost number -->
        <div style="font-family:'Orbitron',monospace;font-size:52px;font-weight:900;
                    color:{accent};opacity:.08;letter-spacing:-.02em;
                    min-width:56px;line-height:1;">{number:02d}</div>
        <!-- colored bar -->
        <div style="width:4px;min-height:48px;border-radius:2px;flex-shrink:0;
                    background:linear-gradient(180deg,{accent},{accent}44);
                    box-shadow:0 0 12px {accent}88;"></div>
        <!-- text stack -->
        <div style="display:flex;flex-direction:column;gap:3px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                      color:{accent}55;letter-spacing:.18em;text-transform:uppercase;">
              {icon}</div>
          <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:700;
                      color:{accent};letter-spacing:.1em;text-transform:uppercase;
                      text-shadow:0 0 20px {accent}88,0 0 40px {accent}33;
                      animation:sec-hdr-pulse 3s ease-in-out infinite;">
              {title}</div>
          {"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:"+accent+"55;letter-spacing:.1em;margin-top:1px;'>"+subtitle+"</div>" if subtitle else ""}
        </div>
      </div>
      <style>@keyframes sec-hdr-pulse{{
        0%,100%{{text-shadow:0 0 18px {accent}77,0 0 36px {accent}33;}}
        50%{{text-shadow:0 0 30px {accent}cc,0 0 60px {accent}55;}}
      }}</style>
    """, unsafe_allow_html=True)

def sec_wrap_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k in ["aqi_result","category","explanation","audio_path","speech_text","df_loaded","selected_row"]:
    if k not in st.session_state: st.session_state[k]=None


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header-wrap">
  <div class="page-header-glow"></div>
  <div class="page-title">◈ AQI INTELLIGENCE SYSTEM</div>
  <div class="page-title-underline"></div>
  <div class="page-tags">
    <span class="page-tag">◈ 18-FEATURE ML</span>
    <span class="page-tag">◈ EXCEL UPLOAD</span>
    <span class="page-tag">◈ LLM ANALYSIS</span>
    <span class="page-tag">◈ 3D VIZ + RADAR</span>
    <span class="page-tag">◈ VOICE ALERT</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
sec_wrap_open("#14b89a", 0, "◈ STEP ONE", "UPLOAD SENSOR DATA",
              "supported: .xlsx · .xls · .csv")

up_col, info_col = st.columns([1.4, 1])
with up_col:
    uploaded = st.file_uploader("Drop your AQI_Input_Template.xlsx here",
                                type=["xlsx","xls","csv"], label_visibility="collapsed")
with info_col:
    st.markdown("""
    <div style="background:rgba(5,14,10,.7);border:1px solid rgba(20,184,154,.16);
                border-left:3px solid #14b89a;border-radius:0 10px 10px 0;
                padding:15px 18px;font-family:'JetBrains Mono',monospace;font-size:12px;
                color:rgba(20,184,154,.65);line-height:2.0;">
        ◈ Upload <span style="color:#14b89a;font-weight:600;">AQI_Input_Template.xlsx</span><br>
        ◈ Or any <span style="color:#14b89a;">.xlsx / .csv</span> with required columns<br>
        ◈ Select a row to analyse from the dropdown<br>
        ◈ Click <span style="color:#14b89a;font-weight:600;">RUN FULL ANALYSIS</span>
    </div>""", unsafe_allow_html=True)
sec_wrap_close()

# ── Load file ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, sheet_name="AQI_INPUT", header=2)
            df = df[df.iloc[:,0].notna()]
        st.session_state.df_loaded = df
    except Exception:
        try:
            uploaded.seek(0)
            df = pd.read_excel(uploaded) if not uploaded.name.endswith(".csv") else pd.read_csv(uploaded)
            st.session_state.df_loaded = df
        except Exception as e:
            st.error(f"Could not read file: {e}")

# ── Data preview + row select ─────────────────────────────────────────────────
if st.session_state.df_loaded is not None:
    df = st.session_state.df_loaded

    sec_wrap_open("#4ab3d4", 0, "◇ STEP TWO", "SELECT SENSOR READING",
                  "preview · row selection")
    st.markdown("""<style>
    [data-testid="stDataFrame"] { border-radius:10px !important; }
    [data-testid="stDataFrame"] table { font-family:'JetBrains Mono',monospace !important;
        font-size:12px !important; background:rgba(5,14,10,.85) !important; }
    [data-testid="stDataFrame"] th { background:rgba(5,18,12,.95) !important;
        color:#14b89a !important; font-size:11px !important; letter-spacing:.07em !important; }
    [data-testid="stDataFrame"] td { color:#4ab3d4 !important; font-size:12px !important; }
    </style>""", unsafe_allow_html=True)
    st.dataframe(df.head(10), width='stretch', height=220)
    row_labels = [f"Row {i+1}  —  {str(df.iloc[i].get('Timestamp', df.index[i]))[:16]}"
                  for i in range(len(df))]
    sel_label = st.selectbox("◈ Select a sensor reading row to analyse:",
                             row_labels, label_visibility="visible")
    sel_idx   = row_labels.index(sel_label)
    row       = df.iloc[sel_idx]
    st.session_state.selected_row = row
    sec_wrap_close()

    # ── Parse row ──────────────────────────────────────────────────────────────
    def safe(key, default=0.0):
        for col in df.columns:
            if key.lower() in col.lower():
                val = row.get(col, default)
                try:
                    f = float(val); return f if not math.isnan(f) else default
                except: return default
        return default

    pm25=safe("PM2.5"); pm10=safe("PM10"); co=safe("CO (ppm)",1.0)
    co2=safe("CO2");    no2=safe("NO2");   so2=safe("SO2")
    o3=safe("O3");      temp=safe("Temp"); hum=safe("Humid")
    wind=safe("Wind_Sp"); pres=safe("Pressure",1013.0); vis=safe("Visib",10.0)
    traf_raw=str(row.get([c for c in df.columns if "traffic" in c.lower()][0]
                 if any("traffic" in c.lower() for c in df.columns) else "Traffic_Density","Low"))
    traf={"Low":25.0,"Medium":55.0,"High":85.0}.get(traf_raw,safe("Traffic",50.0))
    batt=safe("Battery"); dqi=safe("Data_Quality",0.85)
    pm_ratio = round(pm25/pm10,3) if pm10>0 else 0.0
    thi      = round(temp+0.55*(1-hum/100)*(temp-14.5),2)
    co2_co_r = round(co2/co,2) if co>0 else 0.0
    feats18  = [pm25,pm10,co,co2,no2,so2,o3,temp,hum,wind,pres,vis,traf,batt,dqi,pm_ratio,thi,co2_co_r]

    # ── Run button ────────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1.5,1,1.5])
    with btn_col:
        run_btn = st.button("⟶  RUN FULL ANALYSIS", width='stretch')

    if run_btn:
        model   = load_model()
        aqi_val = predict_aqi(model,feats18) if model else round(pm25*1.5+pm10*.5+no2*.3,1)
        if not model: st.warning("Model not found — using formula fallback.")
        st.session_state.aqi_result  = aqi_val
        st.session_state.category    = classify_aqi(aqi_val)
        st.session_state.explanation = None
        st.session_state.audio_path  = None
        st.session_state.speech_text = None
        with st.spinner("⟳  Generating AI analysis via Groq LLaMA 3.1..."):
            try:
                expl = gen_explanation(aqi_val,pm25,pm10,co,co2,no2,so2,o3,temp,hum,wind)
                st.session_state.explanation = expl
                with open("explanation.txt","w") as f: f.write(expl)
            except Exception as e:
                st.error(f"LLM error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.aqi_result is not None:
        AQI=st.session_state.aqi_result; CAT=st.session_state.category
        CLR=aqi_color(CAT); BDGE=aqi_badge(CAT)

        # DANGER BANNER
        if AQI > 150:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(40,8,4,.92),rgba(22,4,2,.97));
                        border:1px solid #e85a3a33;border-left:4px solid #e85a3a;
                        border-radius:10px;padding:14px 22px;margin-bottom:16px;
                        animation:pb 2.2s infinite;">
              <style>@keyframes pb{{0%,100%{{box-shadow:0 0 18px rgba(232,90,58,.12);}}
              50%{{box-shadow:0 0 44px rgba(232,90,58,.35);}}}}</style>
              <span style="font-family:'Orbitron',monospace;font-size:16px;color:#e85a3a;
                           font-weight:700;letter-spacing:.1em;
                           text-shadow:0 0 18px #e85a3a88;">⚠ CRITICAL ALERT</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                           color:#e85a3a88;margin-left:16px;">
                  AQI {AQI} · {CAT.upper()} · AVOID OUTDOORS · WEAR N95 · SEAL WINDOWS
              </span></div>""", unsafe_allow_html=True)

        # ── 01: 3D + GAUGE + METRICS ──────────────────────────────────────────
        sec_wrap_open(CLR, 1, "◈ ANALYSIS", "AQI PREDICTION RESULT",
                      f"predicted score · category · key pollutants")
        v_col, g_col, m_col = st.columns([1,1,1])
        with v_col:
            st.components.v1.html(get_3d_viz(AQI,CAT), height=340, scrolling=False)
        with g_col:
            st.components.v1.html(get_aqi_gauge(AQI,CAT), height=340, scrolling=False)
        with m_col:
            st.markdown(
                neon_metric("AQI SCORE", AQI, "", CLR) +
                f'<div style="text-align:center;margin-top:8px;">'
                f'<span style="font-family:Orbitron,monospace;font-size:11px;'
                f'padding:5px 14px;border-radius:20px;{BDGE};letter-spacing:.1em;">{CAT}</span></div>',
                unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            for cols3, items in [
                (st.columns(3),[("PM2.5",pm25,"µg/m³","#14b89a"),("PM10",pm10,"µg/m³","#e8a443"),("NO₂",no2,"µg/m³","#e8723a")]),
                (st.columns(3),[("CO",co,"ppm","#8a6fe8"),("SO₂",so2,"µg/m³","#e85a3a"),("O₃",o3,"µg/m³","#4ab3d4")]),
            ]:
                for col,(lbl,val,unit,clr) in zip(cols3,items):
                    with col: st.markdown(neon_metric(lbl,val,unit,clr), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sec_wrap_close()

        # ── 02: RADAR ─────────────────────────────────────────────────────────
        sec_wrap_open("#e8a443", 2, "◉ POLLUTANTS", "RADAR ANALYSIS",
                      "normalised concentration · 8 pollutants")
        rc1, rc2 = st.columns([1,1.1])
        with rc1:
            st.components.v1.html(get_radar(pm25,pm10,no2,so2,o3,co,co2,traf), height=400, scrolling=False)
        with rc2:
            bars_html="".join(glowing_bar(n,v,mx,c,u) for n,c,v,mx,u in [
                ("PM2.5","#14b89a",pm25,300,"µg/m³"),("PM10","#e8a443",pm10,400,"µg/m³"),
                ("NO₂","#e8723a",no2,200,"µg/m³"),("SO₂","#8a6fe8",so2,200,"µg/m³"),
                ("O₃","#4ab3d4",o3,200,"µg/m³"),("CO","#e85a3a",co,30,"ppm"),
                ("CO₂","#a3d46a",co2,2000,"ppm"),("Traffic","#e8c43a",traf,100,"idx"),
            ])
            st.markdown(f"<div style='padding-top:20px;'>{bars_html}</div>", unsafe_allow_html=True)
        sec_wrap_close()

        # ── 03: ENVIRONMENT ───────────────────────────────────────────────────
        sec_wrap_open("#4ab3d4", 3, "◆ ENVIRONMENT", "ATMOSPHERIC CONDITIONS",
                      "real-time meteorological readings")
        cols6 = st.columns(6)
        for col,(lbl,clr,val,unit,ico) in zip(cols6,[
            ("TEMPERATURE","#e8723a",f"{temp}","°C","🌡"),
            ("HUMIDITY","#4ab3d4",f"{hum}","%","💧"),
            ("WIND","#14b89a",f"{wind}","m/s","🌬"),
            ("PRESSURE","#8a6fe8",f"{pres}","hPa","◎"),
            ("VISIBILITY","#e8a443",f"{vis}","km","◉"),
            ("TRAFFIC","#e85a3a",f"{traf}","idx","◈"),
        ]):
            with col:
                st.markdown(f"""
                <div style="background:rgba(5,14,10,.65);border:1px solid {clr}22;
                            border-radius:12px;padding:16px 10px;text-align:center;
                            position:relative;overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,{clr}88,transparent);"></div>
                    <div style="font-size:20px;margin-bottom:7px;">{ico}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                        letter-spacing:.11em;color:{clr}55;text-transform:uppercase;margin-bottom:5px;">{lbl}</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:700;
                        color:{clr};text-shadow:0 0 12px {clr}55;">{val}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                        color:{clr}38;margin-top:4px;">{unit}</div>
                </div>""", unsafe_allow_html=True)
        sec_wrap_close()

        # ── 04+05: DERIVED + SENSOR ───────────────────────────────────────────
        d_col, s_col = st.columns(2)
        with d_col:
            sec_wrap_open("#8a6fe8", 4, "◎ COMPUTED", "DERIVED FEATURES",
                          "ratios · indices · computed metrics")
            for lbl,val,clr,desc in [
                ("PM2.5 / PM10 RATIO",pm_ratio,"#8a6fe8","Particle size distribution index"),
                ("TEMP–HUMIDITY INDEX",thi,"#e8723a","Thermal stress indicator"),
                ("CO₂ / CO RATIO",co2_co_r,"#14b89a","Combustion completeness ratio"),
            ]:
                st.markdown(f"""
                <div style="background:rgba(8,4,16,.55);border:1px solid {clr}1e;
                            border-left:3px solid {clr}88;border-radius:0 10px 10px 0;
                            padding:14px 16px;margin-bottom:11px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                    color:{clr}88;letter-spacing:.06em;">{lbl}</div>
                        <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                                    color:{clr};text-shadow:0 0 10px {clr}66;">{val}</div>
                    </div>
                    <div style="font-family:'Exo 2',sans-serif;font-size:13px;
                                color:rgba(160,200,180,.55);">{desc}</div>
                </div>""", unsafe_allow_html=True)
            sec_wrap_close()

        with s_col:
            sec_wrap_open("#e85a3a", 5, "◐ HARDWARE", "SENSOR HEALTH",
                          "battery · data quality index")
            bc,dc = bat_color(batt),dqi_color(dqi)
            for lbl,val,pct,clr,ico in [
                ("BATTERY LEVEL",f"{batt}%",int(batt),bc,"⚡"),
                ("DATA QUALITY INDEX",f"{dqi}",int(dqi*100) if dqi<=1 else int(dqi),dc,"◈"),
            ]:
                st.markdown(f"""
                <div style="background:rgba(14,4,2,.55);border:1px solid {clr}1e;
                            border-radius:12px;padding:18px;margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                    color:{clr}55;">{ico} &nbsp;{lbl}</div>
                        <div style="font-family:'Orbitron',monospace;font-size:28px;font-weight:700;
                                    color:{clr};text-shadow:0 0 14px {clr}77;">{val}</div>
                    </div>
                    <div style="height:10px;background:rgba(5,10,6,.8);border-radius:5px;
                                overflow:hidden;border:1px solid {clr}1e;">
                        <div style="width:{pct}%;height:100%;border-radius:5px;
                            background:linear-gradient(90deg,{clr}55,{clr});
                            box-shadow:0 0 12px {clr}88;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:4px;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{clr}28;">0</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{clr}28;">100</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            sec_wrap_close()

        # ── 06: HEALTH ADVISORY ───────────────────────────────────────────────
        sec_wrap_open("#e8a443", 6, "◑ SAFETY", "HEALTH ADVISORY",
                      "precautions · pollution sources")
        hc1, hc2 = st.columns([1.2,1])
        with hc1:
            for ico,clr,title,desc in [
                ("🔴","#e85a3a","STAY INDOORS","Keep all windows and doors sealed tightly."),
                ("🟠","#e8723a","WEAR N95 MASK","Use N95/N100 rated mask when outdoors."),
                ("🟠","#e8a443","AIR PURIFIER","HEPA filter reduces indoor particulates."),
                ("🟡","#e8c43a","VULNERABLE GROUPS","Children & elderly at significantly higher risk."),
                ("🟢","#14b89a","SEEK MEDICAL HELP","Consult doctor if breathing difficulty worsens."),
            ]:
                st.markdown(f"""
                <div style="display:flex;gap:12px;background:rgba(5,12,8,.5);
                            border:1px solid {clr}1e;border-left:3px solid {clr}88;
                            border-radius:0 10px 10px 0;padding:12px 15px;margin-bottom:9px;">
                    <div style="font-size:16px;flex-shrink:0;padding-top:2px;">{ico}</div>
                    <div>
                        <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                                    color:{clr};letter-spacing:.1em;margin-bottom:3px;
                                    text-shadow:0 0 12px {clr}66;">{title}</div>
                        <div style="font-family:'Exo 2',sans-serif;font-size:13px;
                                    color:rgba(150,210,170,.68);line-height:1.5;">{desc}</div>
                    </div></div>""", unsafe_allow_html=True)
        with hc2:
            st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                           color:rgba(20,184,154,.7);letter-spacing:.12em;margin-bottom:14px;
                           text-shadow:0 0 14px rgba(20,184,154,.4);">
                           ◈ POLLUTION SOURCES</div>""", unsafe_allow_html=True)
            grid="<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>"
            for ico,lbl,clr in [("🚗","Traffic","#e85a3a"),("🏭","Industry","#e8723a"),
                ("🌫","Dust","#e8a443"),("🏗","Construction","#8a6fe8"),
                ("🔥","Burning","#e8c43a"),("⚡","Power Plants","#4ab3d4"),
                ("🚢","Shipping","#14b89a"),("🌾","Agriculture","#a3d46a")]:
                grid+=f"""<div style="background:rgba(5,14,10,.65);border:1px solid {clr}22;
                    border-radius:10px;padding:11px 12px;display:flex;align-items:center;gap:9px;">
                    <span style="font-size:16px;">{ico}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{clr}bb;">{lbl}</span>
                    </div>"""
            grid+="</div>"
            st.markdown(grid, unsafe_allow_html=True)
        sec_wrap_close()

        # ── 07: AI ANALYSIS  — animated typewriter terminal ───────────────────
        sec_wrap_open("#14b89a", 7, "◍ INTELLIGENCE", "AI-GENERATED ANALYSIS",
                      "powered by groq · llama-3.1-8b")
        if st.session_state.explanation:
            html = get_ai_analysis_html(st.session_state.explanation)
            # estimate height by line count
            n_lines = len([l for l in st.session_state.explanation.split("\n") if l.strip()])
            height  = max(400, min(n_lines * 36 + 120, 900))
            st.components.v1.html(html, height=height, scrolling=True)
        else:
            st.markdown("""
            <div style="background:rgba(3,12,8,.85);border:1px solid rgba(20,184,154,.12);
                        border-radius:10px;padding:20px 24px;
                        font-family:'JetBrains Mono',monospace;font-size:13px;
                        color:rgba(20,184,154,.42);font-style:italic;">
                Analysis pending — run the model to generate insights.
                <span style="display:inline-block;width:8px;height:14px;background:#14b89a;
                             vertical-align:middle;margin-left:4px;border-radius:1px;
                             animation:blink .85s step-end infinite;"></span>
                <style>@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0;}}}}</style>
            </div>""", unsafe_allow_html=True)
        sec_wrap_close()

        # ── 08: VOICE ALERT ───────────────────────────────────────────────────
        sec_wrap_open("#8a6fe8", 8, "◈ AUDIO", "VOICE ALERT",
                      "ai-synthesized · gtts · autoplay")
        va1, va2 = st.columns([1,1.4])
        with va1:
            st.components.v1.html(get_waveform("#8a6fe8"), height=80, scrolling=False)
            if st.session_state.explanation:
                if st.button("▶  SYNTHESIZE VOICE ALERT"):
                    with st.spinner("Synthesizing..."):
                        try:
                            stext=gen_voice(st.session_state.explanation)
                            apath=to_mp3(stext)
                            st.session_state.speech_text=stext
                            st.session_state.audio_path=apath
                        except Exception as e:
                            st.error(f"Speech error: {e}")
            af=st.session_state.audio_path or ("alert.mp3" if os.path.exists("alert.mp3") else None)
            if af and os.path.exists(af): play_audio(af)
        with va2:
            if st.session_state.speech_text:
                st.markdown(f"""
                <div style="background:rgba(8,4,16,.7);border:1px solid #8a6fe822;
                            border-left:3px solid #8a6fe8;border-radius:0 12px 12px 0;
                            padding:18px 20px;font-family:'JetBrains Mono',monospace;
                            font-size:14px;color:rgba(200,180,255,.85);line-height:1.9;
                            font-style:italic;">
                    🎙 {st.session_state.speech_text}</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(8,4,16,.45);border:1px dashed rgba(138,111,232,.25);
                            border-radius:12px;padding:18px 20px;
                            font-family:'JetBrains Mono',monospace;font-size:13px;
                            color:rgba(160,130,220,.45);font-style:italic;">
                    Click SYNTHESIZE to generate and play the voice alert...</div>""",
                unsafe_allow_html=True)
        sec_wrap_close()

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:90px 40px;
                background:linear-gradient(135deg,rgba(5,18,12,.65),rgba(3,10,7,.85));
                border:1px dashed rgba(20,184,154,.12);border-radius:18px;
                backdrop-filter:blur(6px);">
        <div style="font-size:52px;margin-bottom:18px;">🌐</div>
        <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                    color:rgba(20,184,154,.5);letter-spacing:.12em;margin-bottom:12px;
                    text-shadow:0 0 24px rgba(20,184,154,.3);">AWAITING INPUT</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                    color:rgba(20,184,154,.38);line-height:2.4;letter-spacing:.06em;">
            Upload the <span style="color:rgba(20,184,154,.7);font-weight:600;">
            AQI_Input_Template.xlsx</span> file above<br>
            Select a sensor reading row<br>
            Click ⟶ RUN FULL ANALYSIS
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:14px 0;border-top:1px solid rgba(20,184,154,.08);
            font-family:'JetBrains Mono',monospace;font-size:11px;
            color:rgba(20,184,154,.28);letter-spacing:.11em;">
    AQI INTELLIGENCE SYSTEM &nbsp;◈&nbsp; EXCEL-DRIVEN &nbsp;◈&nbsp;
    GROQ LLAMA-3.1-8B &nbsp;◈&nbsp; THREE.JS &nbsp;◈&nbsp; GTTS &nbsp;◈&nbsp; SCIKIT-LEARN
</div>""", unsafe_allow_html=True)
