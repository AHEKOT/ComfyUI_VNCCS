import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

const source=readFileSync(new URL('../web/vnccs_common.js',import.meta.url),'utf8');
const code=source.slice(source.indexOf('function positionWizardReveal'),source.indexOf('// Reveals wizard-populated controls'));
for(const [scaleX,scaleY] of [[1,1],[.67,.67],[.5025,.5025],[.5,.8]])test(`wizard snapshot shares target bounds at ${scaleX} × ${scaleY}`,()=>{
    const position=vm.runInNewContext(`${code};positionWizardReveal`,{getComputedStyle:()=>({borderRadius:'8px'})});
    const field={offsetWidth:500,offsetHeight:300,clientLeft:1,clientTop:2,scrollLeft:7,scrollTop:11,
        getBoundingClientRect:()=>({left:100,top:50,width:500*scaleX,height:300*scaleY})};
    const target={getBoundingClientRect:()=>({left:100+21*scaleX,top:50+32*scaleY,width:450*scaleX,height:200*scaleY})};
    const layer={style:{}};position({field,target,layer});
    const left=100+(parseFloat(layer.style.left)+field.clientLeft-field.scrollLeft)*scaleX;
    const top=50+(parseFloat(layer.style.top)+field.clientTop-field.scrollTop)*scaleY;
    const expected=target.getBoundingClientRect();
    assert.ok(Math.abs(left-expected.left)<.001);
    assert.ok(Math.abs(top-expected.top)<.001);
    assert.ok(Math.abs(parseFloat(layer.style.width)*scaleX-expected.width)<.001);
    assert.ok(Math.abs(parseFloat(layer.style.height)*scaleY-expected.height)<.001);
});
