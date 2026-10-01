exec(open('core.py').read())
def jitter_fit(df, design=None):
    # ML fit of offsets(+design) and per-label jitter
    labs=[l for l in LABS if (df.lab==l).any()]
    t,y,e=df.t.values,df.y.values,df.e.values
    L=np.array([[1.0*(l==x) for x in labs] for l in df.lab])
    X=L if design is None else np.hstack([L,design])
    idx=np.array([labs.index(l) for l in df.lab])
    def nll(lj):
        s2=e**2+np.exp(lj)[idx]**2; w=1/s2
        A=X.T@(X*w[:,None]); b=X.T@(w*y); p=np.linalg.solve(A,b); r=y-X@p
        return 0.5*np.sum(r**2*w+np.log(s2))
    best=None
    for x0 in [np.full(len(labs),v) for v in (-1.,0.5,1.5,2.2)]:
        r=minimize(nll,x0,method='L-BFGS-B',bounds=[(-5,4)]*len(labs))
        if best is None or r.fun<best.fun: best=r
    res=best
    return np.exp(res.x), res.fun, idx, L, labs
def periodogram(df, freqs, jit=None, Y=None):
    jj,_,idx,L,labs=jitter_fit(df) if jit is None else (jit,None,*jitter_fit(df)[2:])
    t=df.t.values; s=np.sqrt(df.e.values**2+jj[idx]**2); w=1/s
    Lw=L*w[:,None]; Q,_=np.linalg.qr(Lw); P=np.eye(len(t))-Q@Q.T
    y=(df.y.values if Y is None else Y)
    yw=(y*w[:,None]) if y.ndim==2 else y*w
    yp=P@yw
    out=[]
    for ch in np.array_split(freqs, max(1,len(freqs)//3000)):
        ph=2*np.pi*np.outer(t-t.mean(),ch)
        C=P@(np.cos(ph)*w[:,None]); Sn=P@(np.sin(ph)*w[:,None])
        cc=(C*C).sum(0); ss=(Sn*Sn).sum(0); cs=(C*Sn).sum(0); det=cc*ss-cs**2
        a=C.T@yp; b=Sn.T@yp
        if yp.ndim==1:
            out.append((ss*a*a-2*cs*a*b+cc*b*b)/det)
        else:
            out.append((ss[:,None]*a*a-2*cs[:,None]*a*b+cc[:,None]*b*b)/det[:,None])
    return np.concatenate(out), jj
T=S.t.max()-S.t.min()
freqs=np.arange(1/1000,1/1.05,1/(5*T))
