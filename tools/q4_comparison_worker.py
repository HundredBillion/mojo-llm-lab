"""Experimental scalar/DP4A comparison worker; use run_q4_comparison.py."""

def main():
    import ctypes as C,json,re,sys,statistics
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'tests'))
    from q4_reference import dot_q4_q8
    ptxs=list((repo/'build/q4-dp4a').glob('*.ptx'));assert len(ptxs)==1
    text=ptxs[0].read_text();entries=re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)',text,re.S);assert len(entries)==1
    name,params=entries[0];assert len(re.findall(r'\.param\s+\.u64',params))==3
    cuda=C.CDLL('libcuda.so.1')
    def fn(name,types):
     f=getattr(cuda,name);f.argtypes=types;f.restype=C.c_int
     def call(*args):
      status=f(*args)
      if status:raise RuntimeError(name+' status '+str(status))
     return call
    ptr=C.c_void_p;u64=C.c_uint64;u=C.c_uint;size=C.c_size_t
    init=fn('cuInit',[u]);get=fn('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);flags=fn('cuDevicePrimaryCtxSetFlags',[C.c_int,u]);retain=fn('cuDevicePrimaryCtxRetain',[C.POINTER(ptr),C.c_int]);current=fn('cuCtxSetCurrent',[ptr]);release=fn('cuDevicePrimaryCtxRelease_v2',[C.c_int]);alloc=fn('cuMemAlloc_v2',[C.POINTER(u64),size]);free=fn('cuMemFree_v2',[u64]);htod=fn('cuMemcpyHtoD_v2',[u64,ptr,size]);dtoh=fn('cuMemcpyDtoH_v2',[ptr,u64,size]);load=fn('cuModuleLoad',[C.POINTER(ptr),C.c_char_p]);func=fn('cuModuleGetFunction',[C.POINTER(ptr),ptr,C.c_char_p]);launch=fn('cuLaunchKernel',[ptr,u,u,u,u,u,u,u,ptr,C.POINTER(ptr),C.POINTER(ptr)]);sync=fn('cuCtxSynchronize',[]);unload=fn('cuModuleUnload',[ptr])
    init(0);dev=C.c_int();get(C.byref(dev),0);flags(dev,4);ctx=ptr();retain(C.byref(ctx),dev);current(ctx)
    a=u64();b=u64();activation=u64();module=ptr()
    try:
     data=(repo/'build/q4-matvec/weights.bin').read_bytes();acts=(repo/'build/q4-matvec/activations.bin').read_bytes();rows=len(data)//1440
     expected=[dot_q4_q8(data[row*1440:(row+1)*1440],acts) for row in range(rows)]
     expected=expected*152;data=data*152;rows=9728
     source=C.create_string_buffer(data);output=(C.c_float*(len(expected)+16))(*([123456.0]*(len(expected)+16)))
     activation_host=C.create_string_buffer(acts);alloc(C.byref(activation),len(acts));htod(activation,activation_host,len(acts))
     alloc(C.byref(a),len(data));alloc(C.byref(b),C.sizeof(output));htod(a,source,len(data));htod(b,output,C.sizeof(output))
     load(C.byref(module),str(ptxs[0]).encode());kernel=ptr();func(C.byref(kernel),module,name.encode())
     offset=u64(b.value+8*4);args=(ptr*3)(C.cast(C.byref(a),ptr),C.cast(C.byref(activation),ptr),C.cast(C.byref(offset),ptr))
     launch(kernel,rows,1,1,32,1,1,0,None,args,None);sync();dtoh(output,b,C.sizeof(output))
     actual=list(output);assert actual[:8]==actual[-8:]==[123456.0]*8
     errors=[abs(a-b) for a,b in zip(actual[8:-8],expected)]
     assert all(e<=1e-5+1e-5*abs(v) for e,v in zip(errors,expected)),max(errors)
     graphs=[];executables=[];events=[];modules=[]
     class KernelParams(C.Structure):
      _fields_=[('func',ptr),('gridX',u),('gridY',u),('gridZ',u),('blockX',u),('blockY',u),('blockZ',u),('shared',u),('kernelParams',C.POINTER(ptr)),('extra',C.POINTER(ptr))]
     graph_create=fn('cuGraphCreate',[C.POINTER(ptr),u]);event_create=fn('cuEventCreate',[C.POINTER(ptr),u]);event_node=fn('cuGraphAddEventRecordNode',[C.POINTER(ptr),ptr,C.POINTER(ptr),size,ptr]);kernel_node=fn('cuGraphAddKernelNode',[C.POINTER(ptr),ptr,C.POINTER(ptr),size,C.POINTER(KernelParams)]);instantiate=fn('cuGraphInstantiateWithFlags',[C.POINTER(ptr),ptr,C.c_ulonglong]);graph_launch=fn('cuGraphLaunch',[ptr,ptr]);elapsed=fn('cuEventElapsedTime',[C.POINTER(C.c_float),ptr,ptr])
     arms={}
     for label,directory in [('scalar','q4-matvec'),('dp4a','q4-dp4a')]:
      path=next((repo/'build'/directory).glob('*.ptx'));entry=re.search(r'\.visible\s+\.entry\s+(\w+)',path.read_text()).group(1)
      mod=ptr();load(C.byref(mod),str(path).encode());modules.append(mod);f=ptr();func(C.byref(f),mod,entry.encode())
      start=ptr();end=ptr();event_create(C.byref(start),0);event_create(C.byref(end),0);events.extend([start,end]);g=ptr();graph_create(C.byref(g),0);graphs.append(g)
      node=ptr();event_node(C.byref(node),g,None,0,start)
      params=KernelParams(f,rows,1,1,32,1,1,0,args,None)
      for repeat in range(32):
       deps=(ptr*1)(node);new=ptr();kernel_node(C.byref(new),g,deps,1,C.byref(params));node=new
      deps=(ptr*1)(node);new=ptr();event_node(C.byref(new),g,deps,1,end)
      executable=ptr();instantiate(C.byref(executable),g,0);executables.append(executable);arms[label]=(executable,start,end)
      for _ in range(2):graph_launch(executable,None);sync()
      dtoh(output,b,C.sizeof(output));assert list(output)[8:-8]==expected and list(output)[:8]==list(output)[-8:]==[123456.0]*8
     nv=C.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0;device=ptr();nv.nvmlDeviceGetHandleByIndex_v2.argtypes=[u,C.POINTER(ptr)];assert nv.nvmlDeviceGetHandleByIndex_v2(0,C.byref(device))==0;nv.nvmlDeviceGetClockInfo.argtypes=[ptr,u,C.POINTER(u)]
     samples=[]
     for control in [True,False]:
      for round_index in range(8):
       for label in ['scalar','candidate','candidate','scalar']:
        executable,start,end=arms['scalar' if control or label=='scalar' else 'dp4a'];graph_launch(executable,None);sync();ms=C.c_float();elapsed(C.byref(ms),start,end);clock=u();assert nv.nvmlDeviceGetClockInfo(device,0,C.byref(clock))==0
        samples.append({'control':control,'round':round_index,'arm':label,'gpu_us_per_matvec':ms.value*1000/32,'gpu_clock_mhz':clock.value})
     dtoh(output,b,C.sizeof(output));assert list(output)[8:-8]==expected and list(output)[:8]==list(output)[-8:]==[123456.0]*8
     nv.nvmlShutdown()
     medians={str(control):{label:statistics.median(r['gpu_us_per_matvec'] for r in samples if r['control']==control and r['arm']==label) for label in ['scalar','candidate']} for control in [True,False]}
     control_ratio=medians['True']['scalar']/medians['True']['candidate'];assert .95<=control_ratio<=1.05,control_ratio
     result={'status':'passed','rows':rows,'width':2560,'exact_outputs':True,'canaries':'intact','repetitions_per_graph':32,'samples':samples,'medians_us':medians,'control_ratio':control_ratio,'scalar_over_dp4a':medians['False']['scalar']/medians['False']['candidate'],'scope':'Mojo scalar vs Mojo DP4A, not GGML CUDA timing or whole inference gain'}
     Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
     for executable in executables:fn('cuGraphExecDestroy',[ptr])(executable)
     for graph in graphs:fn('cuGraphDestroy',[ptr])(graph)
     for event in events:fn('cuEventDestroy_v2',[ptr])(event)
     for mod in modules:unload(mod)

    finally:
     if module.value:unload(module)
     if activation.value:free(activation)
     if a.value:free(a)
     if b.value:free(b)
     release(dev)

if __name__ == "__main__":
    main()
