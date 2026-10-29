"""Regression: a missing animation callback timestamp must not blank the arena."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('initial',[
    'lastTick=900;elapsed=0;',
    'lastTick=NaN;elapsed=NaN;',
])
def test_playback_clock_advances_without_a_callback_timestamp(initial):
    node=shutil.which('node')
    if node is None:
        candidate=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        if candidate.is_file():node=str(candidate)
    if node is None:pytest.skip('Node required for JavaScript playback regression')
    html=Path('demo/reference.html').read_text(encoding='utf-8')
    match=re.search(r'function tick\([^)]*\)\s*\{.*?\n\}',html,re.S)
    assert match is not None
    setup="""
let clock=1000,playing=true,lastTick=900,elapsed=0,frame=0,draws=0,scheduled=0;
const data={physics:{dt:.1},times:Array.from({length:100},(_,i)=>i*.1)};
const performance={now:()=>clock};
function $(id){return {value:'2'}}
function requestAnimationFrame(){scheduled++}
function pause(){playing=false}
function draw(){if(!Number.isFinite(frame))throw Error('invalid frame');draws++}
function status(){}
"""
    script=setup+initial+match.group(0)+"""
tick();const first=frame;clock=1200;tick();
console.log(JSON.stringify({first,frame,elapsed,draws,scheduled,playing}));
"""
    result=subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)
    output=json.loads(result.stdout)
    assert output['frame']>output['first']>=0
    assert output['elapsed']>0 and output['draws']==2 and output['scheduled']==2 and output['playing']
