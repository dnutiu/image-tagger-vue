/*!
 * ONNX Runtime Web v1.15.1
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
var _scriptDir,e=(_scriptDir="undefined"!=typeof document&&document.currentScript?document.currentScript.src:void 0,"undefined"!=typeof __filename&&(_scriptDir=_scriptDir||__filename),function(e){function n(){return k.buffer!=C&&B(k.buffer),F}function t(){return k.buffer!=C&&B(k.buffer),P}function r(){return k.buffer!=C&&B(k.buffer),I}function a(){return k.buffer!=C&&B(k.buffer),U}function o(){return k.buffer!=C&&B(k.buffer),Y}var u,i,s;e=e||{},u||(u=void 0!==e?e:{}),u.ready=new Promise((function(e,n){i=e,s=n}));var f,c,l,p,m,d,h=Object.assign({},u),g="./this.program",y=(e,n)=>{throw n},_="object"==typeof window,b="function"==typeof importScripts,v="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node,w=u.ENVIRONMENT_IS_PTHREAD||!1,O="";function T(e){return u.locateFile?u.locateFile(e,O):O+e}if(v){let e;O=b?require("path").dirname(O)+"/":__dirname+"/",d=()=>{m||(p=require("fs"),m=require("path"))},f=function(e,n){return d(),e=m.normalize(e),p.readFileSync(e,n?void 0:"utf8")},l=e=>((e=f(e,!0)).buffer||(e=new Uint8Array(e)),e),c=(e,n,t)=>{d(),e=m.normalize(e),p.readFile(e,(function(e,r){e?t(e):n(r.buffer)}))},1<process.argv.length&&(g=process.argv[1].replace(/\\/g,"/")),process.argv.slice(2),process.on("uncaughtException",(function(e){if(!(e instanceof oe))throw e})),process.on("unhandledRejection",(function(e){throw e})),y=(e,n)=>{if(E)throw process.exitCode=e,n;n instanceof oe||x("exiting due to exception: "+n),process.exit(e)},u.inspect=function(){return"[Emscripten Module object]"};try{e=require("worker_threads")}catch(e){throw console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'),e}global.Worker=e.Worker}else(_||b)&&(b?O=self.location.href:"undefined"!=typeof document&&document.currentScript&&(O=document.currentScript.src),_scriptDir&&(O=_scriptDir),O=0!==O.indexOf("blob:")?O.substr(0,O.replace(/[?#].*/,"").lastIndexOf("/")+1):"",v||(f=e=>{var n=new XMLHttpRequest;return n.open("GET",e,!1),n.send(null),n.responseText},b&&(l=e=>{var n=new XMLHttpRequest;return n.open("GET",e,!1),n.responseType="arraybuffer",n.send(null),new Uint8Array(n.response)}),c=(e,n,t)=>{var r=new XMLHttpRequest;r.open("GET",e,!0),r.responseType="arraybuffer",r.onload=()=>{200==r.status||0==r.status&&r.response?n(r.response):t()},r.onerror=t,r.send(null)}));v&&"undefined"==typeof performance&&(global.performance=require("perf_hooks").performance);var M=console.log.bind(console),S=console.warn.bind(console);v&&(d(),M=e=>p.writeSync(1,e+"\n"),S=e=>p.writeSync(2,e+"\n"));var A,R=u.print||M,x=u.printErr||S;Object.assign(u,h),h=null,u.thisProgram&&(g=u.thisProgram),u.quit&&(y=u.quit),u.wasmBinary&&(A=u.wasmBinary);var E=u.noExitRuntime||!0;"object"!=typeof WebAssembly&&ne("no native wasm support detected");var k,D,C,F,P,I,U,Y,W=!1,H="undefined"!=typeof TextDecoder?new TextDecoder("utf8"):void 0;function j(e,n,t){var r=(n>>>=0)+t;for(t=n;e[t]&&!(t>=r);)++t;if(16<t-n&&e.buffer&&H)return H.decode(e.buffer instanceof SharedArrayBuffer?e.slice(n,t):e.subarray(n,t));for(r="";n<t;){var a=e[n++];if(128&a){var o=63&e[n++];if(192==(224&a))r+=String.fromCharCode((31&a)<<6|o);else{var u=63&e[n++];65536>(a=224==(240&a)?(15&a)<<12|o<<6|u:(7&a)<<18|o<<12|u<<6|63&e[n++])?r+=String.fromCharCode(a):(a-=65536,r+=String.fromCharCode(55296|a>>10,56320|1023&a))}}else r+=String.fromCharCode(a)}return r}function N(e,n){return(e>>>=0)?j(t(),e,n):""}function L(e,n,t,r){if(!(0<r))return 0;var a=t>>>=0;r=t+r-1;for(var o=0;o<e.length;++o){var u=e.charCodeAt(o);if(55296<=u&&57343>=u&&(u=65536+((1023&u)<<10)|1023&e.charCodeAt(++o)),127>=u){if(t>=r)break;n[t++>>>0]=u}else{if(2047>=u){if(t+1>=r)break;n[t++>>>0]=192|u>>6}else{if(65535>=u){if(t+2>=r)break;n[t++>>>0]=224|u>>12}else{if(t+3>=r)break;n[t++>>>0]=240|u>>18,n[t++>>>0]=128|u>>12&63}n[t++>>>0]=128|u>>6&63}n[t++>>>0]=128|63&u}}return n[t>>>0]=0,t-a}function q(e){for(var n=0,t=0;t<e.length;++t){var r=e.charCodeAt(t);127>=r?n++:2047>=r?n+=2:55296<=r&&57343>=r?(n+=4,++t):n+=3}return n}function B(e){C=e,u.HEAP8=F=new Int8Array(e),u.HEAP16=new Int16Array(e),u.HEAP32=I=new Int32Array(e),u.HEAPU8=P=new Uint8Array(e),u.HEAPU16=new Uint16Array(e),u.HEAPU32=U=new Uint32Array(e),u.HEAPF32=new Float32Array(e),u.HEAPF64=Y=new Float64Array(e)}w&&(C=u.buffer);var G=u.INITIAL_MEMORY||16777216;if(w)k=u.wasmMemory,C=u.buffer;else if(u.wasmMemory)k=u.wasmMemory;else if(!((k=new WebAssembly.Memory({initial:G/65536,maximum:65536,shared:!0})).buffer instanceof SharedArrayBuffer))throw x("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"),v&&console.log("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)"),Error("bad memory");k&&(C=k.buffer),G=C.byteLength,B(C);var z,J=[],K=[],Q=[];function V(){var e=u.preRun.shift();J.unshift(e)}var X,Z=0,$=null,ee=null;function ne(e){throw w?postMessage({cmd:"onAbort",arg:e}):u.onAbort&&u.onAbort(e),x(e="Aborted("+e+")"),W=!0,e=new WebAssembly.RuntimeError(e+". Build with -sASSERTIONS for more info."),s(e),e}function te(){return X.startsWith("data:application/octet-stream;base64,")}function re(){var e=X;try{if(e==X&&A)return new Uint8Array(A);if(l)return l(e);throw"both async and sync fetching of the wasm failed"}catch(e){ne(e)}}X="ort-wasm-threaded.wasm",te()||(X=T(X));var ae={};function oe(e){this.name="ExitStatus",this.message="Program terminated with exit("+e+")",this.status=e}function ue(e){(e=ce.La[e])||ne(),ce.Xa(e)}function ie(e){var n=ce.lb();if(!n)return 6;ce.Ra.push(n),ce.La[e.Ka]=n,n.Ka=e.Ka;var t={cmd:"run",start_routine:e.pb,arg:e.ib,pthread_ptr:e.Ka};return n.Qa=()=>{t.time=performance.now(),n.postMessage(t,e.vb)},n.loaded&&(n.Qa(),delete n.Qa),0}function se(e){if(w)return We(1,1,e);E||(ce.qb(),u.onExit&&u.onExit(e),W=!0),y(e,new oe(e))}function fe(e,n){if(!n&&w)throw pe(e),"unwind";se(e)}var ce={Oa:[],Ra:[],$a:[],La:{},Ua:function(){w&&ce.mb()},xb:function(){},mb:function(){ce.receiveObjectTransfer=ce.ob,ce.threadInitTLS=ce.Za,ce.setExitStatus=ce.Ya,E=!1},Ya:function(){},qb:function(){for(var e of Object.values(ce.La))ce.Xa(e);for(e of ce.Oa)e.terminate();ce.Oa=[]},Xa:function(e){var n=e.Ka;delete ce.La[n],ce.Oa.push(e),ce.Ra.splice(ce.Ra.indexOf(e),1),e.Ka=0,fn(n)},ob:function(){},Za:function(){ce.$a.forEach((e=>e()))},nb:function(e,n){e.onmessage=t=>{var r=(t=t.data).cmd;if(e.Ka&&(ce.kb=e.Ka),t.targetThread&&t.targetThread!=tn()){var a=ce.La[t.yb];a?a.postMessage(t,t.transferList):x('Internal error! Worker sent a message "'+r+'" to target pthread '+t.targetThread+", but that thread no longer exists!")}else"processProxyingQueue"===r?Ce(t.queue):"spawnThread"===r?ie(t):"cleanupThread"===r?ue(t.thread):"killThread"===r?(t=t.thread,r=ce.La[t],delete ce.La[t],r.terminate(),fn(t),ce.Ra.splice(ce.Ra.indexOf(r),1),r.Ka=0):"cancelThread"===r?ce.La[t.thread].postMessage({cmd:"cancel"}):"loaded"===r?(e.loaded=!0,n&&n(e),e.Qa&&(e.Qa(),delete e.Qa)):"print"===r?R("Thread "+t.threadId+": "+t.text):"printErr"===r?x("Thread "+t.threadId+": "+t.text):"alert"===r?alert("Thread "+t.threadId+": "+t.text):"setimmediate"===t.target?e.postMessage(t):"onAbort"===r?u.onAbort&&u.onAbort(t.arg):r&&x("worker sent an unknown command "+r);ce.kb=void 0},e.onerror=e=>{throw x("worker sent an error! "+e.filename+":"+e.lineno+": "+e.message),e},v&&(e.on("message",(function(n){e.onmessage({data:n})})),e.on("error",(function(n){e.onerror(n)})),e.on("detachedExit",(function(){}))),e.postMessage({cmd:"load",urlOrBlob:u.mainScriptUrlOrBlob||_scriptDir,wasmMemory:k,wasmModule:D})},hb:function(){var e=T("ort-wasm-threaded.worker.js");ce.Oa.push(new Worker(e))},lb:function(){return 0==ce.Oa.length&&(ce.hb(),ce.nb(ce.Oa[0])),ce.Oa.pop()}};function le(e){for(;0<e.length;)e.shift()(u)}function pe(e){if(w)return We(2,0,e);try{fe(e)}catch(e){e instanceof oe||"unwind"==e||y(1,e)}}u.PThread=ce,u.establishStackSpace=function(){var e=tn(),n=r()[e+44>>2>>>0];e=r()[e+48>>2>>>0],ln(n,n-e),mn(n)};var me,de,he=[];function ge(e){this.Pa=e-24,this.gb=function(e){a()[this.Pa+4>>2>>>0]=e},this.cb=function(e){a()[this.Pa+8>>2>>>0]=e},this.eb=function(){r()[this.Pa>>2>>>0]=0},this.bb=function(){n()[this.Pa+12>>0>>>0]=0},this.fb=function(){n()[this.Pa+13>>0>>>0]=0},this.Ua=function(e,n){this.ab(),this.gb(e),this.cb(n),this.eb(),this.bb(),this.fb()},this.ab=function(){a()[this.Pa+16>>2>>>0]=0}}function ye(e,n,t,r){return w?We(3,1,e,n,t,r):_e(e,n,t,r)}function _e(e,n,t,r){if("undefined"==typeof SharedArrayBuffer)return x("Current environment does not support SharedArrayBuffer, pthreads are not available!"),6;var a=[];return w&&0===a.length?ye(e,n,t,r):(e={pb:t,Ka:e,ib:r,vb:a},w?(e.wb="spawnThread",postMessage(e,a),0):ie(e))}function be(e,n,t){return w?We(4,1,e,n,t):0}function ve(e,n){if(w)return We(5,1,e,n)}function we(e,n){if(w)return We(6,1,e,n)}function Oe(e,n,t){if(w)return We(7,1,e,n,t)}function Te(e,n,t){return w?We(8,1,e,n,t):0}function Me(e,n){if(w)return We(9,1,e,n)}function Se(e,n,t){if(w)return We(10,1,e,n,t)}function Ae(e,n,t,r){if(w)return We(11,1,e,n,t,r)}function Re(e,n,t,r){if(w)return We(12,1,e,n,t,r)}function xe(e,n,t,r){if(w)return We(13,1,e,n,t,r)}function Ee(e){if(w)return We(14,1,e)}function ke(e,n){if(w)return We(15,1,e,n)}function De(e,n,t){if(w)return We(16,1,e,n,t)}function Ce(e){Atomics.store(r(),e>>2,1),tn()&&sn(e),Atomics.compareExchange(r(),e>>2,1,0)}function Fe(e){return a()[e>>>2]+4294967296*r()[e+4>>>2]}function Pe(e,n,t,r,a,o){return w?We(17,1,e,n,t,r,a,o):-52}function Ie(e,n,t,r,a,o){if(w)return We(18,1,e,n,t,r,a,o)}function Ue(e){var t=q(e)+1,r=rn(t);return r&&L(e,n(),r,t),r}function Ye(e,n,t){function o(e){return(e=e.toTimeString().match(/\(([A-Za-z ]+)\)$/))?e[1]:"GMT"}if(w)return We(19,1,e,n,t);var u=(new Date).getFullYear(),i=new Date(u,0,1),s=new Date(u,6,1);u=i.getTimezoneOffset();var f=s.getTimezoneOffset(),c=Math.max(u,f);r()[e>>2>>>0]=60*c,r()[n>>2>>>0]=Number(u!=f),e=o(i),n=o(s),e=Ue(e),n=Ue(n),f<u?(a()[t>>2>>>0]=e,a()[t+4>>2>>>0]=n):(a()[t>>2>>>0]=n,a()[t+4>>2>>>0]=e)}function We(e,n){var t=arguments.length-2,r=arguments;return function(e){var n=pn();return e=e(),mn(n),e}((()=>{for(var a=dn(8*t),u=a>>3,i=0;i<t;i++){var s=r[2+i];o()[u+i>>>0]=s}return un(e,t,a,n)}))}u.invokeEntryPoint=function(e,n){var t=he[e];t||(e>=he.length&&(he.length=e+1),he[e]=t=z.get(e)),e=t(n),E?ce.Ya(e):cn(e)},u.executeNotifiedProxyingQueue=Ce,de=v?()=>{var e=process.hrtime();return 1e3*e[0]+e[1]/1e6}:w?()=>performance.now()-u.__performance_now_clock_drift:()=>performance.now();var He,je=[],Ne={};function Le(){if(!He){var e,n={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"==typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:g||"./this.program"};for(e in Ne)void 0===Ne[e]?delete n[e]:n[e]=Ne[e];var t=[];for(e in n)t.push(e+"="+n[e]);He=t}return He}function qe(e,t){if(w)return We(20,1,e,t);var r=0;return Le().forEach((function(o,u){var i=t+r;for(u=a()[e+4*u>>2>>>0]=i,i=0;i<o.length;++i)n()[u++>>0>>>0]=o.charCodeAt(i);n()[u>>0>>>0]=0,r+=o.length+1})),0}function Be(e,n){if(w)return We(21,1,e,n);var t=Le();a()[e>>2>>>0]=t.length;var r=0;return t.forEach((function(e){r+=e.length+1})),a()[n>>2>>>0]=r,0}function Ge(e){return w?We(22,1,e):52}function ze(e,n,t,r){return w?We(23,1,e,n,t,r):52}function Je(e,n,t,r,a){return w?We(24,1,e,n,t,r,a):70}var Ke=[null,[],[]];function Qe(e,n,r,o){if(w)return We(25,1,e,n,r,o);for(var u=0,i=0;i<r;i++){var s=a()[n>>2>>>0],f=a()[n+4>>2>>>0];n+=8;for(var c=0;c<f;c++){var l=t()[s+c>>>0],p=Ke[e];0===l||10===l?((1===e?R:x)(j(p,0)),p.length=0):p.push(l)}u+=f}return a()[o>>2>>>0]=u,0}function Ve(e){return 0==e%4&&(0!=e%100||0==e%400)}var Xe=[31,29,31,30,31,30,31,31,30,31,30,31],Ze=[31,28,31,30,31,30,31,31,30,31,30,31];function $e(e,t,a,o){function u(e,n,t){for(e="number"==typeof e?e.toString():e||"";e.length<n;)e=t[0]+e;return e}function i(e,n){return u(e,n,"0")}function s(e,n){function t(e){return 0>e?-1:0<e?1:0}var r;return 0===(r=t(e.getFullYear()-n.getFullYear()))&&0===(r=t(e.getMonth()-n.getMonth()))&&(r=t(e.getDate()-n.getDate())),r}function f(e){switch(e.getDay()){case 0:return new Date(e.getFullYear()-1,11,29);case 1:return e;case 2:return new Date(e.getFullYear(),0,3);case 3:return new Date(e.getFullYear(),0,2);case 4:return new Date(e.getFullYear(),0,1);case 5:return new Date(e.getFullYear()-1,11,31);case 6:return new Date(e.getFullYear()-1,11,30)}}function c(e){var n=e.Ma;for(e=new Date(new Date(e.Na+1900,0,1).getTime());0<n;){var t=e.getMonth(),r=(Ve(e.getFullYear())?Xe:Ze)[t];if(!(n>r-e.getDate())){e.setDate(e.getDate()+n);break}n-=r-e.getDate()+1,e.setDate(1),11>t?e.setMonth(t+1):(e.setMonth(0),e.setFullYear(e.getFullYear()+1))}return t=new Date(e.getFullYear()+1,0,4),n=f(new Date(e.getFullYear(),0,4)),t=f(t),0>=s(n,e)?0>=s(t,e)?e.getFullYear()+1:e.getFullYear():e.getFullYear()-1}var l=r()[o+40>>2>>>0];for(var p in o={tb:r()[o>>2>>>0],sb:r()[o+4>>2>>>0],Sa:r()[o+8>>2>>>0],Va:r()[o+12>>2>>>0],Ta:r()[o+16>>2>>>0],Na:r()[o+20>>2>>>0],Ja:r()[o+24>>2>>>0],Ma:r()[o+28>>2>>>0],zb:r()[o+32>>2>>>0],rb:r()[o+36>>2>>>0],ub:l?N(l):""},a=N(a),l={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"})a=a.replace(new RegExp(p,"g"),l[p]);var m="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),d="January February March April May June July August September October November December".split(" ");for(p in l={"%a":function(e){return m[e.Ja].substring(0,3)},"%A":function(e){return m[e.Ja]},"%b":function(e){return d[e.Ta].substring(0,3)},"%B":function(e){return d[e.Ta]},"%C":function(e){return i((e.Na+1900)/100|0,2)},"%d":function(e){return i(e.Va,2)},"%e":function(e){return u(e.Va,2," ")},"%g":function(e){return c(e).toString().substring(2)},"%G":function(e){return c(e)},"%H":function(e){return i(e.Sa,2)},"%I":function(e){return 0==(e=e.Sa)?e=12:12<e&&(e-=12),i(e,2)},"%j":function(e){for(var n=0,t=0;t<=e.Ta-1;n+=(Ve(e.Na+1900)?Xe:Ze)[t++]);return i(e.Va+n,3)},"%m":function(e){return i(e.Ta+1,2)},"%M":function(e){return i(e.sb,2)},"%n":function(){return"\n"},"%p":function(e){return 0<=e.Sa&&12>e.Sa?"AM":"PM"},"%S":function(e){return i(e.tb,2)},"%t":function(){return"\t"},"%u":function(e){return e.Ja||7},"%U":function(e){return i(Math.floor((e.Ma+7-e.Ja)/7),2)},"%V":function(e){var n=Math.floor((e.Ma+7-(e.Ja+6)%7)/7);if(2>=(e.Ja+371-e.Ma-2)%7&&n++,n)53==n&&(4==(t=(e.Ja+371-e.Ma)%7)||3==t&&Ve(e.Na)||(n=1));else{n=52;var t=(e.Ja+7-e.Ma-1)%7;(4==t||5==t&&Ve(e.Na%400-1))&&n++}return i(n,2)},"%w":function(e){return e.Ja},"%W":function(e){return i(Math.floor((e.Ma+7-(e.Ja+6)%7)/7),2)},"%y":function(e){return(e.Na+1900).toString().substring(2)},"%Y":function(e){return e.Na+1900},"%z":function(e){var n=0<=(e=e.rb);return e=Math.abs(e)/60,(n?"+":"-")+String("0000"+(e/60*100+e%60)).slice(-4)},"%Z":function(e){return e.ub},"%%":function(){return"%"}},a=a.replace(/%%/g,"\0\0"),l)a.includes(p)&&(a=a.replace(new RegExp(p,"g"),l[p](o)));return p=function(e){var n=Array(q(e)+1);return L(e,n,0,n.length),n}(a=a.replace(/\0\0/g,"%")),p.length>t?0:(function(e,t){n().set(e,t>>>0)}(p,e),p.length-1)}ce.Ua();var en=[null,se,pe,ye,be,ve,we,Oe,Te,Me,Se,Ae,Re,xe,Ee,ke,De,Pe,Ie,Ye,qe,Be,Ge,ze,Je,Qe],nn={b:function(e){return rn(e+24)+24},c:function(e,n,t){throw new ge(e).Ua(n,t),e},L:function(e){an(e,!b,1,!_),ce.Za()},l:function(e){w?postMessage({cmd:"cleanupThread",thread:e}):ue(e)},D:_e,i:be,R:ve,z:we,B:Oe,T:Te,P:Me,I:Se,O:Ae,p:Re,A:xe,x:Ee,Q:ke,y:De,r:function(){},j:function(){ne("To use dlopen, you need enable dynamic linking, see https://github.com/emscripten-core/emscripten/wiki/Linking")},s:function(){ne("To use dlopen, you need enable dynamic linking, see https://github.com/emscripten-core/emscripten/wiki/Linking")},q:function(){return Date.now()},E:function(){return 2097152},V:function(){return!0},F:function(e,n,t,r){if(e==n)setTimeout((()=>Ce(r)));else if(w)postMessage({targetThread:e,cmd:"processProxyingQueue",queue:r});else{if(!(e=ce.La[e]))return;e.postMessage({cmd:"processProxyingQueue",queue:r})}return 1},K:function(){return-1},W:function(e,n){e=new Date(1e3*Fe(e)),r()[n>>2>>>0]=e.getUTCSeconds(),r()[n+4>>2>>>0]=e.getUTCMinutes(),r()[n+8>>2>>>0]=e.getUTCHours(),r()[n+12>>2>>>0]=e.getUTCDate(),r()[n+16>>2>>>0]=e.getUTCMonth(),r()[n+20>>2>>>0]=e.getUTCFullYear()-1900,r()[n+24>>2>>>0]=e.getUTCDay(),e=(e.getTime()-Date.UTC(e.getUTCFullYear(),0,1,0,0,0,0))/864e5|0,r()[n+28>>2>>>0]=e},X:function(e,n){e=new Date(1e3*Fe(e)),r()[n>>2>>>0]=e.getSeconds(),r()[n+4>>2>>>0]=e.getMinutes(),r()[n+8>>2>>>0]=e.getHours(),r()[n+12>>2>>>0]=e.getDate(),r()[n+16>>2>>>0]=e.getMonth(),r()[n+20>>2>>>0]=e.getFullYear()-1900,r()[n+24>>2>>>0]=e.getDay();var t=new Date(e.getFullYear(),0,1),a=(e.getTime()-t.getTime())/864e5|0;r()[n+28>>2>>>0]=a,r()[n+36>>2>>>0]=-60*e.getTimezoneOffset(),a=new Date(e.getFullYear(),6,1).getTimezoneOffset(),e=0|(a!=(t=t.getTimezoneOffset())&&e.getTimezoneOffset()==Math.min(t,a)),r()[n+32>>2>>>0]=e},Y:function(e){var n=new Date(r()[e+20>>2>>>0]+1900,r()[e+16>>2>>>0],r()[e+12>>2>>>0],r()[e+8>>2>>>0],r()[e+4>>2>>>0],r()[e>>2>>>0],0),t=r()[e+32>>2>>>0],a=n.getTimezoneOffset(),o=new Date(n.getFullYear(),0,1),u=new Date(n.getFullYear(),6,1).getTimezoneOffset(),i=o.getTimezoneOffset(),s=Math.min(i,u);return 0>t?r()[e+32>>2>>>0]=Number(u!=i&&s==a):0<t!=(s==a)&&(u=Math.max(i,u),n.setTime(n.getTime()+6e4*((0<t?s:u)-a))),r()[e+24>>2>>>0]=n.getDay(),t=(n.getTime()-o.getTime())/864e5|0,r()[e+28>>2>>>0]=t,r()[e>>2>>>0]=n.getSeconds(),r()[e+4>>2>>>0]=n.getMinutes(),r()[e+8>>2>>>0]=n.getHours(),r()[e+12>>2>>>0]=n.getDate(),r()[e+16>>2>>>0]=n.getMonth(),n.getTime()/1e3|0},G:Pe,H:Ie,Z:function e(n,t,r){e.jb||(e.jb=!0,Ye(n,t,r))},d:function(){ne("")},m:function(){if(!v&&!b){var e="Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread";me||(me={}),me[e]||(me[e]=1,v&&(e="warning: "+e),x(e))}},w:function(){return 4294901760},f:de,S:function(e,n,r){t().copyWithin(e>>>0,n>>>0,n+r>>>0)},g:function(){return v?require("os").cpus().length:navigator.hardwareConcurrency},J:function(e,n,t){je.length=n,t>>=3;for(var r=0;r<n;r++)je[r]=o()[t+r>>>0];return(0>e?ae[-e-1]:en[e]).apply(null,je)},v:function(e){var n=t().length;if((e>>>=0)<=n||4294901760<e)return!1;for(var r=1;4>=r;r*=2){var a=n*(1+.2/r);a=Math.min(a,e+100663296);var o=Math;a=Math.max(e,a),o=o.min.call(o,4294901760,a+(65536-a%65536)%65536);e:{try{k.grow(o-C.byteLength+65535>>>16),B(k.buffer);var u=1;break e}catch(e){}u=void 0}if(u)return!0}return!1},U:function(){throw"unwind"},M:qe,N:Be,k:fe,h:Ge,o:ze,t:Je,n:Qe,u:function e(t,r){e.Wa||(e.Wa=function(){if("object"==typeof crypto&&"function"==typeof crypto.getRandomValues){var e=new Uint8Array(1);return()=>(crypto.getRandomValues(e),e[0])}if(v)try{var n=require("crypto");return()=>n.randomBytes(1)[0]}catch(e){}return()=>ne("randomDevice")}());for(var a=0;a<r;a++)n()[t+a>>0>>>0]=e.Wa();return 0},a:k||u.wasmMemory,C:$e,e:function(e,n,t,r){return $e(e,n,t,r)}};!function(){function e(e,n){u.asm=e.exports,ce.$a.push(u.asm.wa),z=u.asm.za,K.unshift(u.asm._),D=n,w||(Z--,u.monitorRunDependencies&&u.monitorRunDependencies(Z),0==Z&&(null!==$&&(clearInterval($),$=null),ee&&(e=ee,ee=null,e())))}function n(n){e(n.instance,n.module)}function t(e){return function(){if(!A&&(_||b)){if("function"==typeof fetch&&!X.startsWith("file://"))return fetch(X,{credentials:"same-origin"}).then((function(e){if(!e.ok)throw"failed to load wasm binary file at '"+X+"'";return e.arrayBuffer()})).catch((function(){return re()}));if(c)return new Promise((function(e,n){c(X,(function(n){e(new Uint8Array(n))}),n)}))}return Promise.resolve().then((function(){return re()}))}().then((function(e){return WebAssembly.instantiate(e,r)})).then((function(e){return e})).then(e,(function(e){x("failed to asynchronously prepare wasm: "+e),ne(e)}))}var r={a:nn};if(w||(Z++,u.monitorRunDependencies&&u.monitorRunDependencies(Z)),u.instantiateWasm)try{return u.instantiateWasm(r,e)}catch(e){return x("Module.instantiateWasm callback failed with error: "+e),!1}(A||"function"!=typeof WebAssembly.instantiateStreaming||te()||X.startsWith("file://")||v||"function"!=typeof fetch?t(n):fetch(X,{credentials:"same-origin"}).then((function(e){return WebAssembly.instantiateStreaming(e,r).then(n,(function(e){return x("wasm streaming compile failed: "+e),x("falling back to ArrayBuffer instantiation"),t(n)}))}))).catch(s)}(),u.___wasm_call_ctors=function(){return(u.___wasm_call_ctors=u.asm._).apply(null,arguments)},u._OrtInit=function(){return(u._OrtInit=u.asm.$).apply(null,arguments)},u._OrtCreateSessionOptions=function(){return(u._OrtCreateSessionOptions=u.asm.aa).apply(null,arguments)},u._OrtAppendExecutionProvider=function(){return(u._OrtAppendExecutionProvider=u.asm.ba).apply(null,arguments)},u._OrtAddSessionConfigEntry=function(){return(u._OrtAddSessionConfigEntry=u.asm.ca).apply(null,arguments)},u._OrtReleaseSessionOptions=function(){return(u._OrtReleaseSessionOptions=u.asm.da).apply(null,arguments)},u._OrtCreateSession=function(){return(u._OrtCreateSession=u.asm.ea).apply(null,arguments)},u._OrtReleaseSession=function(){return(u._OrtReleaseSession=u.asm.fa).apply(null,arguments)},u._OrtGetInputCount=function(){return(u._OrtGetInputCount=u.asm.ga).apply(null,arguments)},u._OrtGetOutputCount=function(){return(u._OrtGetOutputCount=u.asm.ha).apply(null,arguments)},u._OrtGetInputName=function(){return(u._OrtGetInputName=u.asm.ia).apply(null,arguments)},u._OrtGetOutputName=function(){return(u._OrtGetOutputName=u.asm.ja).apply(null,arguments)},u._OrtFree=function(){return(u._OrtFree=u.asm.ka).apply(null,arguments)},u._OrtCreateTensor=function(){return(u._OrtCreateTensor=u.asm.la).apply(null,arguments)},u._OrtGetTensorData=function(){return(u._OrtGetTensorData=u.asm.ma).apply(null,arguments)},u._OrtReleaseTensor=function(){return(u._OrtReleaseTensor=u.asm.na).apply(null,arguments)},u._OrtCreateRunOptions=function(){return(u._OrtCreateRunOptions=u.asm.oa).apply(null,arguments)},u._OrtAddRunConfigEntry=function(){return(u._OrtAddRunConfigEntry=u.asm.pa).apply(null,arguments)},u._OrtReleaseRunOptions=function(){return(u._OrtReleaseRunOptions=u.asm.qa).apply(null,arguments)},u._OrtRun=function(){return(u._OrtRun=u.asm.ra).apply(null,arguments)},u._OrtEndProfiling=function(){return(u._OrtEndProfiling=u.asm.sa).apply(null,arguments)};var tn=u._pthread_self=function(){return(tn=u._pthread_self=u.asm.ta).apply(null,arguments)},rn=u._malloc=function(){return(rn=u._malloc=u.asm.ua).apply(null,arguments)};u._free=function(){return(u._free=u.asm.va).apply(null,arguments)},u.__emscripten_tls_init=function(){return(u.__emscripten_tls_init=u.asm.wa).apply(null,arguments)};var an=u.__emscripten_thread_init=function(){return(an=u.__emscripten_thread_init=u.asm.xa).apply(null,arguments)};u.__emscripten_thread_crashed=function(){return(u.__emscripten_thread_crashed=u.asm.ya).apply(null,arguments)};var on,un=u._emscripten_run_in_main_runtime_thread_js=function(){return(un=u._emscripten_run_in_main_runtime_thread_js=u.asm.Aa).apply(null,arguments)},sn=u.__emscripten_proxy_execute_task_queue=function(){return(sn=u.__emscripten_proxy_execute_task_queue=u.asm.Ba).apply(null,arguments)},fn=u.__emscripten_thread_free_data=function(){return(fn=u.__emscripten_thread_free_data=u.asm.Ca).apply(null,arguments)},cn=u.__emscripten_thread_exit=function(){return(cn=u.__emscripten_thread_exit=u.asm.Da).apply(null,arguments)},ln=u._emscripten_stack_set_limits=function(){return(ln=u._emscripten_stack_set_limits=u.asm.Ea).apply(null,arguments)},pn=u.stackSave=function(){return(pn=u.stackSave=u.asm.Fa).apply(null,arguments)},mn=u.stackRestore=function(){return(mn=u.stackRestore=u.asm.Ga).apply(null,arguments)},dn=u.stackAlloc=function(){return(dn=u.stackAlloc=u.asm.Ha).apply(null,arguments)};function hn(){function e(){if(!on&&(on=!0,u.calledRun=!0,!W)&&(w||le(K),i(u),u.onRuntimeInitialized&&u.onRuntimeInitialized(),!w)){if(u.postRun)for("function"==typeof u.postRun&&(u.postRun=[u.postRun]);u.postRun.length;){var e=u.postRun.shift();Q.unshift(e)}le(Q)}}if(!(0<Z))if(w)i(u),w||le(K),postMessage({cmd:"loaded"});else{if(u.preRun)for("function"==typeof u.preRun&&(u.preRun=[u.preRun]);u.preRun.length;)V();le(J),0<Z||(u.setStatus?(u.setStatus("Running..."),setTimeout((function(){setTimeout((function(){u.setStatus("")}),1),e()}),1)):e())}}if(u.___cxa_is_pointer_type=function(){return(u.___cxa_is_pointer_type=u.asm.Ia).apply(null,arguments)},u.UTF8ToString=N,u.stringToUTF8=function(e,n,r){return L(e,t(),n,r)},u.lengthBytesUTF8=q,u.keepRuntimeAlive=function(){return E},u.wasmMemory=k,u.stackSave=pn,u.stackRestore=mn,u.stackAlloc=dn,u.ExitStatus=oe,u.PThread=ce,ee=function e(){on||hn(),on||(ee=e)},u.preInit)for("function"==typeof u.preInit&&(u.preInit=[u.preInit]);0<u.preInit.length;)u.preInit.pop()();return hn(),e.ready});"object"==typeof exports&&"object"==typeof module?module.exports=e:"function"==typeof define&&define.amd?define([],(function(){return e})):"object"==typeof exports&&(exports.ortWasmThreaded=e);